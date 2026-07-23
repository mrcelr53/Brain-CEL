# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Marcel Rinder

import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

INPUT_COLOR = "C0"
OUTPUT_COLOR = "C1"

def load(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        sys.exit(f"No spikes found in {csv_path}")
    return (np.atleast_1d(data["time_ms"]),
            np.atleast_1d(data["neuron_id"]),
            np.atleast_1d(data["layer"]).astype(int))

def population_rate(times, t_max, bin_ms, n_neurons):
    edges = np.arange(0.0, t_max + bin_ms, bin_ms)
    counts, _ = np.histogram(times, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rate_hz = counts / (bin_ms * 1e-3) / max(n_neurons, 1)   # spikes/s/neuron
    return centers, rate_hz

def load_weights(weights_path):
    data = np.genfromtxt(weights_path, delimiter=",", names=True)
    return np.atleast_1d(data["w_init"]), np.atleast_1d(data["w_final"])

def load_potential(potential_path):
    data = np.genfromtxt(potential_path, delimiter=",", names=True)
    return np.atleast_1d(data["time_ms"]), np.atleast_1d(data["potential"])


def visualize(csv_path, weights_path=None, potential_path=None,
              output_path="quickstart_raster.png", bin_ms=10.0):
    times, ids, layers = load(csv_path)
    t_max = float(times.max())

    fig = plt.figure(figsize=(14, 8), dpi=150, layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1.1], height_ratios=[3, 1, 1])
    ax_r = fig.add_subplot(gs[0, 0])                  # raster
    ax_h = fig.add_subplot(gs[1, 0], sharex=ax_r)     # population rate
    ax_v = fig.add_subplot(gs[2, 0], sharex=ax_r)     # membrane potential
    ax_w = fig.add_subplot(gs[:, 1])                  # STDP weight change

    # --- raster ---
    for lyr, color, name in ((0, INPUT_COLOR, "Input"), (1, OUTPUT_COLOR, "Output")):
        m = layers == lyr
        ax_r.scatter(times[m], ids[m], s=2, c=color, marker="|",
                     linewidths=1.5, label=f"{name} ({int(m.sum())} spikes)")
    ax_r.set_ylabel("neuron id")
    ax_r.set_title("Spike raster")
    ax_r.legend(loc="upper right", markerscale=6, framealpha=0.9)
    ax_r.set_ylim(ids.min() - 5, ids.max() + 5)

    # --- population rate ---
    for lyr, color, name in ((0, INPUT_COLOR, "Input"), (1, OUTPUT_COLOR, "Output")):
        m = layers == lyr
        if not m.any():
            continue
        n_neurons = int(np.unique(ids[m]).size)          # active neurons in this layer
        centers, rate = population_rate(times[m], t_max, bin_ms, n_neurons)
        ax_h.plot(centers, rate, color=color, lw=1.5, label=name)
        ax_h.set_title("Average firing rate", fontsize=9, loc="left")
    ax_h.set_ylabel(f"rate [Hz]")
    ax_h.set_xlim(0, t_max)
    ax_h.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax_h.tick_params(labelbottom=False)

    # --- membrane potential of a single output neuron ---
    if potential_path and Path(potential_path).exists():
        vt, vm = load_potential(potential_path)
        ax_v.plot(vt, vm, color=OUTPUT_COLOR, lw=0.8)
        ax_v.set_ylabel("voltage [mV]")
        ax_v.set_title("Membrane potential (neuron id 500)", fontsize=9, loc="left")
    else:
        ax_v.text(0.5, 0.5, "no potential file", ha="center", va="center",
                  transform=ax_v.transAxes, color="0.6")
    ax_v.set_xlabel("time [ms]")
    ax_v.set_xlim(0, t_max)
    ax_v.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)

    # --- STDP weight change ---
    if weights_path and Path(weights_path).exists():
        w_init, w_final = load_weights(weights_path)
        init = float(np.mean(w_init))
        ax_w.hist(w_final, bins=60, color=OUTPUT_COLOR, alpha=0.85)
        ax_w.axvline(init, color="0.2", ls="--", lw=1.5,
                     label=f"initial ({init:.3f})")
        ax_w.axvline(float(np.mean(w_final)), color=OUTPUT_COLOR, lw=2.0,
                     label=f"final mean ({np.mean(w_final):.3f})")
        ax_w.set_xlabel("synaptic weight")
        ax_w.set_ylabel("number of synapses")
        nsyn = f"{len(w_final)//1000}k" if len(w_final) < 1e8 else f"{len(w_final)//1000000}M"
        ax_w.set_title(f"STDP weight change\n({nsyn} synapses)")
        ax_w.legend(loc="upper right", fontsize=9, framealpha=0.9)
    else:
        ax_w.set_axis_off()
        ax_w.text(0.5, 0.5, "no weights file", ha="center", va="center",
                  transform=ax_w.transAxes, color="0.6")

    out = Path(output_path)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"Saved raster to {out.resolve()}")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    return fig


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "quickstart_spikes.csv"
    weights_path = sys.argv[2] if len(sys.argv) > 2 else "quickstart_weights.csv"
    potential_path = sys.argv[3] if len(sys.argv) > 3 else "quickstart_potential.csv"
    visualize(csv_path, weights_path, potential_path)


if __name__ == "__main__":
    main()
