# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Your Name

import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec

from matplotlib import rc
rc('font', **{'family': 'serif'})


STRATEGY_LABELS = {
    'forward-triggered': 'FT-STDP',
    'bidirectionally-triggered': 'Bi-STDP',
}

def parse_log_file(filepath):
    results = {}
    current_label = None

    header_re = re.compile(r'^Sweep\b')
    data_re = re.compile(
        r'pre-rate=(?P<pre>[0-9.]+)Hz\s*\|\s*post-rate=(?P<post>[0-9.]+)Hz'
        r'.*?\brtf=(?P<rtf>[0-9.]+)'
    )

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if header_re.match(line):
                # Identify which strategy this sweep belongs to
                current_label = None
                for token, label in STRATEGY_LABELS.items():
                    if token in line:
                        current_label = label
                        break
                if current_label is None:
                    m = re.search(r'\|\s*(.+?)(?:\s*\||$)', line)
                    current_label = m.group(1).strip() if m else 'Unknown'

                results.setdefault(current_label, [])
                continue

            if current_label is None:
                continue

            m = data_re.search(line)
            if m:
                results[current_label].append({
                    'layer1_fire': int(float(m.group('pre'))),
                    'layer2_fire': int(float(m.group('post'))),
                    'rtf': float(m.group('rtf')),
                })

    return results

def aggregate_trials(results_list):
    df = pd.DataFrame(results_list)
    grouped = df.groupby(['layer1_fire', 'layer2_fire']).agg(
        rtf_mean=('rtf', 'mean'),
        rtf_std=('rtf', 'std'),
        n_trials=('rtf', 'count')
    ).reset_index()
    grouped['rtf_std'] = grouped['rtf_std'].fillna(0)
    return grouped

def create_contour_visualization(data_dict, output_path=None, interpolation_resolution=250):
    strategies = list(data_dict.keys())

    all_l1 = set()
    all_l2 = set()
    for grouped in data_dict.values():
        all_l1.update(grouped['layer1_fire'].unique())
        all_l2.update(grouped['layer2_fire'].unique())

    layer1_rates = sorted(all_l1)
    layer2_rates = sorted(all_l2)

    print(f"\nLayer 1 fire rates: {layer1_rates}")
    print(f"Layer 2 fire rates: {layer2_rates}")

    all_rtf_values = []
    for grouped in data_dict.values():
        all_rtf_values.extend(grouped['rtf_mean'].values)
    vmin = min(all_rtf_values)
    vmax = max(all_rtf_values)
    vcenter = 1.0

    if vcenter <= vmin:
        vcenter = vmin + 0.01
    if vcenter >= vmax:
        vcenter = vmax - 0.01

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    contour_levels = np.linspace(vmin * 0.99, vmax * 1.01, 25)
    key_levels = [0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
    key_levels = [l for l in key_levels if vmin <= l <= vmax]

    n_strats = len(strategies)
    fig = plt.figure(figsize=(4.5 * n_strats + 1, 4))
    gs = GridSpec(1, n_strats + 1,
                  width_ratios=[1] * n_strats + [0.05],
                  wspace=0.08)

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_strats)]
    cbar_ax = fig.add_subplot(gs[0, n_strats])

    l1_fine = np.linspace(min(layer1_rates), max(layer1_rates), interpolation_resolution)
    l2_fine = np.linspace(min(layer2_rates), max(layer2_rates), interpolation_resolution)
    L1_grid, L2_grid = np.meshgrid(l1_fine, l2_fine)

    for idx, (name, grouped) in enumerate(data_dict.items()):
        ax = axes[idx]

        points = grouped[['layer1_fire', 'layer2_fire']].values
        values = grouped['rtf_mean'].values

        try:
            Z = griddata(points, values, (L1_grid, L2_grid), method='cubic')
        except Exception:
            Z = griddata(points, values, (L1_grid, L2_grid), method='linear')

        Z_nearest = griddata(points, values, (L1_grid, L2_grid), method='nearest')
        Z = np.where(np.isnan(Z), Z_nearest, Z)

        cf = ax.contourf(L1_grid, L2_grid, Z, levels=contour_levels, cmap='RdYlGn_r',
                         norm=norm, extend='both')

        ax.contour(L1_grid, L2_grid, Z, levels=contour_levels, colors='black',
                   linewidths=0.5, alpha=0.5)

        other_levels = [l for l in key_levels if l != 1.0]
        cs_key = ax.contour(L1_grid, L2_grid, Z, levels=other_levels,
                            colors='black', linewidths=1.2)
        midpoints = get_all_contour_midpoints(cs_key)
        if midpoints:
            ax.clabel(cs_key, inline=True, fontsize=10, fmt='%.2f', manual=midpoints)

        cs_one = ax.contour(L1_grid, L2_grid, Z, levels=[1.0],
                            colors='black', linewidths=2.8)
        center_one = get_contour_midpoint(cs_one, level_idx=0)
        if center_one:
            ax.clabel(cs_one, levels=[1.0], manual=[center_one],
                      fmt='%.2f', fontsize=11, inline=True)

        ax.set_xlabel('Presynaptic Firing Rate [Hz]', fontsize=13)
        ax.set_ylabel('Postsynaptic Firing Rate [Hz]' if idx == 0 else '', fontsize=13)
        ax.set_title('(a) ' + name if idx == 0 else '(b) ' + name, fontsize=15)
        ax.set_xlim(min(layer1_rates), max(layer1_rates))
        ax.set_ylim(min(layer2_rates), max(layer2_rates))

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.)
            spine.set_color('black')

        ax.set_xticks(layer1_rates)
        ax.set_yticks(layer2_rates)
        if idx > 0:
            ax.set_yticklabels([])

    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    cb_ticks = [0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
    cb_ticks = [l for l in cb_ticks if vmin <= l <= vmax]
    cbar.set_ticks(cb_ticks)
    cbar.set_ticklabels([f"{l:g}" for l in cb_ticks])
    cbar.set_label('RTF', fontsize=13)
    cbar.outline.set_linewidth(1.5)
    fig.subplots_adjust(bottom=0.17)

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white',
                    edgecolor='black', pad_inches=0.15)
        print(f"\nVisualization saved to: {output_path}")

    return fig

def get_all_contour_midpoints(contour_set):
    midpoints = []
    if hasattr(contour_set, 'allsegs'):
        for i in range(len(contour_set.levels)):
            segs = contour_set.allsegs[i]
            valid_segs = [s for s in segs if len(s) > 0]
            if valid_segs:
                longest = max(valid_segs, key=lambda s: len(s))
                mid = len(longest) // 2
                midpoints.append((longest[mid, 0], longest[mid, 1]))
    else:
        for i in range(len(contour_set.levels)):
            paths = contour_set.collections[i].get_paths()
            valid = [p for p in paths if len(p.vertices) > 0]
            if valid:
                longest = max(valid, key=lambda p: len(p.vertices))
                v = longest.vertices
                midpoints.append((v[len(v) // 2, 0], v[len(v) // 2, 1]))
    return midpoints

def get_contour_midpoint(contour_set, level_idx=0):
    if hasattr(contour_set, 'allsegs'):
        segs = contour_set.allsegs[level_idx]
        valid = [s for s in segs if len(s) > 0]
        if not valid:
            return None
        longest = max(valid, key=lambda s: len(s))
        mid = len(longest) // 2
        return longest[mid, 0], longest[mid, 1]
    else:
        paths = contour_set.collections[level_idx].get_paths()
        valid = [p for p in paths if len(p.vertices) > 0]
        if not valid:
            return None
        longest = max(valid, key=lambda p: len(p.vertices))
        v = longest.vertices
        return v[len(v) // 2, 0], v[len(v) // 2, 1]


def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'braincel_bench.txt'

    raw = parse_log_file(log_file)

    data_dict = {}
    for label, rows in raw.items():
        grouped = aggregate_trials(rows)
        data_dict[label] = grouped

    output_path = './benchmark_contour.pdf'
    create_contour_visualization(data_dict, output_path)
    plt.show()


if __name__ == '__main__':
    main()