# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Marcel Rinder

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc

rc('font', **{'family': 'serif'})

def load_stdp_data(filepath='stdp_timing.csv'):
    df = pd.read_csv(filepath)
    if 'dt_ms' not in df.columns or 'dw' not in df.columns:
        df = pd.read_csv(filepath, header=None, names=['dt_ms', 'dw'])
    return df


def create_stdp_visualization(csv_path='stdp_timing.csv', output_path=None):
    df = load_stdp_data(csv_path)
    dt = df['dt_ms'].values
    dw = df['dw'].values

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.plot(dt, dw, color=(0.2,0.2,0.2), linewidth=3.2, zorder=3, label='$\Delta w$')
    ax.fill_between(dt, 0, np.maximum(dw, 0),
                    where=dw >= 0,
                    color='#2ca02c', alpha=0.35, zorder=2,
                    label='LTP (Potentiation)')
    ax.fill_between(dt, 0, np.minimum(dw, 0),
                    where=dw <= 0,
                    color='#d62728', alpha=0.35, zorder=2,
                    label='LTD (Depression)')

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.axhline(y=0, color='black', linewidth=1.1, alpha=0.85, zorder=1)
    ax.set_xlabel('$\Delta t$ [ms]', fontsize=14, fontweight='medium')
    ax.set_ylabel('$\Delta w$', fontsize=14, fontweight='medium')
    ax.set_title('STDP Window', fontsize=16, pad=20)
    ax.set_xlim(dt.min() - 2, dt.max() + 2)
    ax.set_ylim(min(dw.min() * 1.08, -0.05), max(dw.max() * 1.08, 0.05))
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=False)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.1, length=5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    return fig

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'stdp_timing.csv'
    fig = create_stdp_visualization(csv_file, output_path='./stdp_window.png')
    plt.show()

if __name__ == '__main__':
    main()