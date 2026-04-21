# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Marcel Rinder

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib import rc
rc('font', **{'family': 'serif'})

BRAINCEL_FILE = sys.argv[1] if len(sys.argv) > 1 else "braincel_bench_4k-80k.txt"
NEST_FILE     = sys.argv[2] if len(sys.argv) > 2 else "nest_bench_n5k_s4m_t4.txt"
OUT_FILE      = sys.argv[3] if len(sys.argv) > 3 else "bench_figure.pdf"


def matchvalue(pattern, line, cast=float, group=1):
    m = re.search(pattern, line)
    return cast(m.group(group)) if m else None

def parse_neuron_sweep(text):
    result = {}
    current_label = None

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Sweep Neurons"):
            m = re.search(r'label=(\S+)', line)
            current_label = m.group(1) if m else "unknown"
            result[current_label] = {}
            continue

        if current_label is None or "n1/2=" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        # expect: syn=XM | n1/2=N | total_n=T | density=D | build-et=... | sim-et=... | speedup=... | rtf=...
        if len(parts) < 8:
            continue

        n_half   = matchvalue(r'n1/2=(\d+)', parts[1], int)
        syn_m    = matchvalue(r'syn=([\d.]+)M', parts[0])
        bm       = matchvalue(r'([\d.]+)s', parts[4])
        bs       = matchvalue(r'±\s*([\d.]+)', parts[4])
        sm       = matchvalue(r'([\d.]+)s', parts[5])
        ss       = matchvalue(r'±\s*([\d.]+)', parts[5])
        speedup  = matchvalue(r'speedup=([\d.]+)', parts[6])
        rtf      = matchvalue(r'rtf=([\d.]+)', parts[7])

        if n_half is None:
            continue

        result[current_label][n_half] = dict(
            syn_m=syn_m, build_mean=bm, build_std=bs,
            sim_mean=sm, sim_std=ss, speedup=speedup, rtf=rtf
        )
    return result
def parse_synapse_sweep(text):
    result = {}
    current_label = None

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Sweep Synapses"):
            m = re.search(r'label=(\S+)', line)
            current_label = m.group(1) if m else "unknown"
            result[current_label] = {}
            continue

        if current_label is None or "outdegree=" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        # expect: syn=XM | outdegree=N/neuron | build-et=... | sim-et=... | speedup=... | rtf=...
        if len(parts) < 6:
            continue

        syn_m   = matchvalue(r'syn=([\d.]+)M', parts[0])
        outdeg  = matchvalue(r'outdegree=(\d+)', parts[1], int)
        bm      = matchvalue(r'([\d.]+)s', parts[2])
        bs      = matchvalue(r'±\s*([\d.]+)', parts[2])
        sm      = matchvalue(r'([\d.]+)s', parts[3])
        ss      = matchvalue(r'±\s*([\d.]+)', parts[3])
        speedup = matchvalue(r'speedup=([\d.]+)', parts[4])
        rtf     = matchvalue(r'rtf=([\d.]+)', parts[5])

        if syn_m is None:
            continue

        result[current_label][syn_m] = dict(
            outdegree=outdeg, build_mean=bm, build_std=bs,
            sim_mean=sm, sim_std=ss, speedup=speedup, rtf=rtf
        )
    return result


with open(BRAINCEL_FILE) as f:
    bc_text = f.read()
with open(NEST_FILE) as f:
    nest_text = f.read()

bc_neurons   = parse_neuron_sweep(bc_text)
bc_synapses  = parse_synapse_sweep(bc_text)
nest_neurons  = parse_neuron_sweep(nest_text)
nest_synapses = parse_synapse_sweep(nest_text)

RATES = [3, 5, 10, 20]

RATE_COLORS_NEST      = ['#aaaaaa', '#888888', '#555555', '#222222']
RATE_COLORS_BC_GPU    = ['#a8d8a8', '#74b974', '#3a9a3a', '#1a6e1a']
RATE_COLORS_BC_CPU    = ['#a8c8e8', '#6aa0cc', '#3070aa', '#10458a']

RATE_LABELS = [f'$r_{{bf}}$ = {r} Hz' for r in RATES]

fig = plt.figure(figsize=(18, 8))
fig.subplots_adjust(wspace=0.35, hspace=0.45)

gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.9])

neuron_axes  = [fig.add_subplot(gs[0, c]) for c in range(4)]
syn_axes     = [fig.add_subplot(gs[1, c]) for c in range(4)]
ax_ncap      = fig.add_subplot(gs[0, 4])
ax_scap      = fig.add_subplot(gs[1, 4])


def neuron_xy(sweep_dict, label, field='rtf'):
    d = sweep_dict.get(label, {})
    xs = sorted(d.keys())
    ys = [d[x][field] for x in xs]
    return [2 * x for x in xs], ys   # total neurons = 2 * n_half

def synapse_xy(sweep_dict, label, field='rtf'):
    d = sweep_dict.get(label, {})
    xs = sorted(d.keys())
    ys = [d[x][field] for x in xs]
    return xs, ys

for idx, (ax, rate) in enumerate(zip(neuron_axes, RATES)):
    letter = chr(ord('a') + idx)
    ax.set_title(f'({letter}) $r_{{bf}}$ = {rate} Hz', fontsize=9)
    ax.set_xlabel('Number of Neurons', fontsize=8)
    ax.set_ylabel('Realtime factor' if idx == 0 else '', fontsize=8)
    ax.axhline(1, color='black', linestyle='--', linewidth=0.8, label='RTF=1')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0,5)

    # NEST
    nest_label = f'NEST_n_{rate}hz'
    xs, ys = neuron_xy(nest_neurons, nest_label)
    if xs:
        ax.plot(xs, ys, 'o-', color=RATE_COLORS_NEST[idx],
                markersize=3, linewidth=1.2, label='NEST')

    # BrainCEL GPU
    bc_gpu_label = f'GPU_n_{rate}hz'
    xs, ys = neuron_xy(bc_neurons, bc_gpu_label)
    if xs:
        ax.plot(xs, ys, 's-', color=RATE_COLORS_BC_GPU[idx],
                markersize=3, linewidth=1.2, label='Brain-CEL (GPU)')

    # BrainCEL CPU
    bc_cpu_label = f'CPU_n_{rate}hz'
    xs, ys = neuron_xy(bc_neurons, bc_cpu_label)
    if xs:
        ax.plot(xs, ys, '^--', color=RATE_COLORS_BC_CPU[idx],
                markersize=3, linewidth=1.0, label='Brain-CEL (CPU)')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))

    if idx == 0:
        ax.legend(fontsize=6, loc='upper left')


for idx, (ax, rate) in enumerate(zip(syn_axes, RATES)):
    letter = chr(ord('f') + idx)
    ax.set_title(f'({letter}) $r_{{bf}}$ = {rate} Hz', fontsize=9)
    ax.set_xlabel('Number of Synapses', fontsize=8)
    ax.set_ylabel('Realtime factor' if idx == 0 else '', fontsize=8)
    ax.axhline(1, color='black', linestyle='--', linewidth=0.8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0,8)

    nest_label   = f'NEST_s_{rate}hz'
    bc_gpu_label = f'GPU_s_{rate}hz'
    bc_cpu_label = f'CPU_s_{rate}hz'

    xs, ys = synapse_xy(nest_synapses, nest_label)
    if xs:
        ax.plot(xs, ys, 'o-', color=RATE_COLORS_NEST[idx],
                markersize=3, linewidth=1.2, label='NEST')

    xs, ys = synapse_xy(bc_synapses, bc_gpu_label)
    if xs:
        ax.plot(xs, ys, 's-', color=RATE_COLORS_BC_GPU[idx],
                markersize=3, linewidth=1.2, label='Brain-CEL (GPU)')

    xs, ys = synapse_xy(bc_synapses, bc_cpu_label)
    if xs:
        ax.plot(xs, ys, '^--', color=RATE_COLORS_BC_CPU[idx],
                markersize=3, linewidth=1.0, label='Brain-CEL (CPU)')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{x:.1f}M' if x >= 1 else f'{x*1000:.0f}k'))

    # if idx == 0:
    #     ax.legend(fontsize=6, loc='upper right')

def cap_at_rtf1(sweep_dict, label, x_getter, field='rtf'):
    d = sweep_dict.get(label, {})
    xs_raw = sorted(d.keys())
    if not xs_raw:
        return None
    xs = np.array([x_getter(x) for x in xs_raw], dtype=float)
    ys = np.array([d[x][field] for x in xs_raw], dtype=float)
    # find first crossing above 1
    for i in range(len(ys) - 1):
        if ys[i] <= 1.0 <= ys[i+1] or ys[i] >= 1.0 >= ys[i+1]:
            # linear interpolation
            frac = (1.0 - ys[i]) / (ys[i+1] - ys[i])
            return xs[i] + frac * (xs[i+1] - xs[i])
    return xs[-1] if ys[-1] < 1.0 else xs[0]

bar_groups   = ['NEST', 'Brain-CEL\n(GPU)', 'Brain-CEL\n(CPU)']
bar_x        = np.arange(len(RATES))
bar_width    = 0.22
offsets      = [-1, 0, 1]

for gi, (group, color_set, n_prefix, n_getter) in enumerate([
    ('NEST',        RATE_COLORS_NEST,   nest_neurons,  lambda x: 2*x),
    ('GPU',         RATE_COLORS_BC_GPU, bc_neurons,    lambda x: 2*x),
    ('CPU',         RATE_COLORS_BC_CPU, bc_neurons,    lambda x: 2*x),
]):
    caps = []
    for rate in RATES:
        if group == 'NEST':
            lbl = f'NEST_n_{rate}hz'
            d   = nest_neurons
        elif group == 'GPU':
            lbl = f'GPU_n_{rate}hz'
            d   = bc_neurons
        else:
            lbl = f'CPU_n_{rate}hz'
            d   = bc_neurons
        cap = cap_at_rtf1(d, lbl, lambda x: 2*x)
        caps.append(cap if cap is not None else 0)

    ax_ncap.bar(bar_x + offsets[gi] * bar_width,
                caps, bar_width,
                color=color_set, label=bar_groups[gi])

ax_ncap.set_title('(e) Neuron Cap for RTF$\\approx$1', fontsize=9)
ax_ncap.set_xlabel('Rate', fontsize=8)
ax_ncap.set_ylabel('Number of Neurons', fontsize=8)
ax_ncap.set_xticks(bar_x)
ax_ncap.set_xticklabels([f'{r} Hz' for r in RATES], fontsize=7)
ax_ncap.tick_params(labelsize=7)
ax_ncap.legend(fontsize=6)
ax_ncap.grid(axis='y', alpha=0.25)
ax_ncap.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, p: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))


for gi, (group, color_set) in enumerate([
    ('NEST', RATE_COLORS_NEST),
    ('GPU',  RATE_COLORS_BC_GPU),
    ('CPU',  RATE_COLORS_BC_CPU),
]):
    caps = []
    for rate in RATES:
        if group == 'NEST':
            lbl = f'NEST_s_{rate}hz'
            d   = nest_synapses
        elif group == 'GPU':
            lbl = f'GPU_s_{rate}hz'
            d   = bc_synapses
        else:
            lbl = f'CPU_s_{rate}hz'
            d   = bc_synapses
        cap = cap_at_rtf1(d, lbl, lambda x: x)
        caps.append(cap if cap is not None else 0)

    ax_scap.bar(bar_x + offsets[gi] * bar_width,
                caps, bar_width,
                color=color_set, label=bar_groups[gi])

ax_scap.set_title('(j) Synapse Cap for RTF$\\approx$1', fontsize=9)
ax_scap.set_xlabel('Rate', fontsize=8)
ax_scap.set_ylabel('Number of Synapses', fontsize=8)
ax_scap.set_xticks(bar_x)
ax_scap.set_xticklabels([f'{r} Hz' for r in RATES], fontsize=7)
ax_scap.tick_params(labelsize=7)
ax_scap.legend(fontsize=6)
ax_scap.grid(axis='y', alpha=0.25)
ax_scap.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, p: f'{x:.1f}M' if x >= 1 else f'{x*1000:.0f}k'))

plt.savefig(OUT_FILE, bbox_inches='tight', dpi=150)
plt.show()