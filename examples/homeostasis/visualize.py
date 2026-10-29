# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Marcel Rinder

import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Okabe-Ito (colour-blind safe)
E_COLOR = "#D55E00"     # excitatory
I_COLOR = "#0072B2"     # inhibitory
IN_COLOR = "0.55"       # input
NET_COLOR = "#009E73"   # net drive
# static runs: greys, iSTDP runs: greens (light -> dark = small -> large g0)
STATIC_SHADES = ["#BDBDBD", "#8C8C8C", "#4D4D4D"]
PLASTIC_SHADES = ["#7FD1B9", "#009E73", "#005A41"]

STEADY_MS = 10000.0     # final window used for "end" statistics


def load(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def smooth(y, n=5):
    """Centered moving average over n bins (edges padded)."""
    return np.convolve(np.pad(y, n // 2, mode="edge"), np.ones(n) / n, mode="valid")


def style(ax, title):
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def schematic(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    nodes = {"Input": (1.3, 3, IN_COLOR), "E": (5, 3, E_COLOR), "I": (8.7, 3, I_COLOR)}
    for name, (x, y, c) in nodes.items():
        ax.add_patch(plt.Circle((x, y), 0.95, color=c, alpha=0.9))
        ax.text(x, y, name, ha="center", va="center", color="white", fontsize=11, weight="bold")
    arrow = dict(arrowstyle="-|>", lw=2, mutation_scale=16)
    ax.annotate("", (4.0, 3.15), (2.3, 3.15), arrowprops={**arrow, "color": IN_COLOR})
    ax.text(3.15, 3.55, "STDP", ha="center", fontsize=9, weight="bold")
    ax.annotate("", (7.7, 3.4), (6.0, 3.4), arrowprops={**arrow, "color": E_COLOR,
                "connectionstyle": "arc3,rad=-0.35"})
    ax.text(6.85, 4.55, "static (+)", ha="center", fontsize=9)
    ax.annotate("", (6.0, 2.6), (7.7, 2.6), arrowprops={**arrow, "color": I_COLOR,
                "arrowstyle": "-[", "connectionstyle": "arc3,rad=-0.35"})
    ax.text(6.85, 1.3, "(−)  static  or  iSTDP\ninitial weight $g_0$", ha="center",
            va="top", fontsize=9)
    ax.set_title("A  Network", fontsize=10, loc="left")


def visualize(out_path="homeostasis_summary.png"):
    runs = load("homeostasis_runs.csv")
    series = load("homeostasis_series.csv")
    spikes = load("homeostasis_spikes.csv")
    pot = load("homeostasis_potential.csv")
    weights = load("homeostasis_weights.csv")

    t_max = float(series["time_ms"].max())
    e_ids = np.unique(weights["target"][weights["inhibitory"] == 1])   # E neurons
    i_ids = np.unique(spikes["neuron_id"][spikes["pop"] == 2])

    # per-run style: grey/dashed = static, green/solid = iSTDP, shade = g0
    info = {}
    for mode in (0, 1):
        sel = sorted((r for r in runs if int(r["plastic"]) == mode), key=lambda r: r["g_inh"])
        shades = PLASTIC_SHADES if mode else STATIC_SHADES
        for k, r in enumerate(sel):
            info[int(r["run"])] = dict(label=str(r["label"]), g0=float(r["g_inh"]), plastic=bool(mode),
                                       color=shades[min(k, len(shades) - 1)],
                                       ls="-" if mode else "--")
    static_ids = [r for r in info if not info[r]["plastic"]]
    plastic_ids = [r for r in info if info[r]["plastic"]]
    # representative pair with the same (smallest) initial weight
    ref_s = min(static_ids, key=lambda r: info[r]["g0"]) if static_ids else None
    ref_p = min(plastic_ids, key=lambda r: info[r]["g0"]) if plastic_ids else None

    def run_series(r):
        return series[series["run"] == r]

    def end_mean(r, key):
        s = run_series(r)
        return float(np.mean(s[key][s["time_ms"] >= t_max - STEADY_MS]))

    fig = plt.figure(figsize=(16, 15), dpi=150, layout="constrained")
    grid = fig.add_gridspec(4, 3, height_ratios=[1.1, 1, 1, 0.9])
    fig.suptitle("Plastic inhibition (iSTDP) drives excitatory activity to the same equilibrium; "
                 "static inhibition does not", fontsize=14, weight="bold")

    # --- A: network schematic ---
    schematic(fig.add_subplot(grid[0, 0]))

    # --- B: E rate over time, all runs ---
    ax = fig.add_subplot(grid[0, 1:])
    for r, st in info.items():
        s = run_series(r)
        ax.plot(s["time_ms"] / 1000, smooth(s["rate_E"]), color=st["color"], ls=st["ls"],
                lw=1.8, label=st["label"])
    if plastic_ids:
        target = np.mean([end_mean(r, "rate_E") for r in plastic_ids])
        ax.axhline(target, color=NET_COLOR, lw=6, alpha=0.15)
        ax.text(0.995, target, f" iSTDP equilibrium ≈ {target:.1f} Hz", color=NET_COLOR,
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9)
    style(ax, "B  Excitatory population rate over time")
    ax.set_xlabel("time [s]"); ax.set_ylabel("E rate [Hz]")
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)

    # --- C: mean I->E weight ---
    ax = fig.add_subplot(grid[1, 0])
    for r, st in info.items():
        s = run_series(r)
        ax.plot(s["time_ms"] / 1000, s["w_inh"], color=st["color"], ls=st["ls"], lw=1.6)
    if plastic_ids:
        w_eq = np.mean([end_mean(r, "w_inh") for r in plastic_ids])
        ax.axhline(w_eq, color=NET_COLOR, lw=6, alpha=0.15)
        ax.text(0.995, w_eq, f"equilibrium ≈ {w_eq:.2f} ", color=NET_COLOR,
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9)
    style(ax, "C  Mean I→E weight (inhibition)")
    ax.set_xlabel("time [s]"); ax.set_ylabel("mean weight")
    ax.set_yscale("log")

    # --- D: mean Input->E weight ---
    ax = fig.add_subplot(grid[1, 1])
    for r, st in info.items():
        s = run_series(r)
        ax.plot(s["time_ms"] / 1000, s["w_exc"], color=st["color"], ls=st["ls"], lw=1.6)
    style(ax, "D  Mean Input→E weight (excitatory STDP)")
    ax.set_xlabel("time [s]"); ax.set_ylabel("mean weight")

    # --- E: drive balance ---
    ax = fig.add_subplot(grid[1, 2])
    if ref_p is not None:
        s = run_series(ref_p)
        t = s["time_ms"] / 1000
        ax.plot(t, smooth(s["exc_drive"]), color=E_COLOR, lw=1.8, label="excitatory")
        ax.plot(t, smooth(s["inh_drive"]), color=I_COLOR, lw=1.8, label="inhibitory")
        ax.plot(t, smooth(s["exc_drive"] - s["inh_drive"]), color=NET_COLOR, lw=2.0,
                label="net = exc − inh")
    if ref_s is not None:
        s = run_series(ref_s)
        ax.plot(s["time_ms"] / 1000, smooth(s["exc_drive"] - s["inh_drive"]), color="0.4",
                lw=1.6, ls="--", label=f"net, {info[ref_s]['label']}")
    style(ax, f"E  Synaptic drive per E neuron ({info[ref_p]['label'] if ref_p is not None else ''})")
    ax.set_xlabel("time [s]"); ax.set_ylabel("summed weight / s")
    ax.legend(fontsize=8)

    # --- F: final E rate vs initial inhibitory weight ---
    ax = fig.add_subplot(grid[2, 0])
    for ids, name, c, m, ls in ((static_ids, "static", "0.4", "s", "--"),
                                (plastic_ids, "iSTDP", NET_COLOR, "o", "-")):
        if not ids:
            continue
        g0 = [info[r]["g0"] for r in ids]
        ax.plot(g0, [end_mean(r, "rate_E") for r in ids], marker=m, ls=ls, color=c, lw=2,
                ms=7, label=name)
    style(ax, f"F  E rate in the last {STEADY_MS / 1000:.0f} s vs. initial weight")
    ax.set_xlabel(r"initial I→E weight $g_0$"); ax.set_ylabel("E rate [Hz]")
    ax.set_xscale("log"); ax.set_ylim(bottom=0)
    g_all = sorted({st["g0"] for st in info.values()})
    ax.set_xticks(g_all, [f"{g:g}" for g in g_all]); ax.minorticks_off()
    ax.legend(fontsize=8)

    # per-neuron statistics
    def per_neuron(r):
        w = weights[weights["run"] == r]
        lo = int(e_ids.min())
        n = len(e_ids)
        exc = np.bincount(w["target"][w["inhibitory"] == 0] - lo,
                          weights=w["w_final"][w["inhibitory"] == 0], minlength=n)[:n]
        inh = np.bincount(w["target"][w["inhibitory"] == 1] - lo,
                          weights=w["w_final"][w["inhibitory"] == 1], minlength=n)[:n]
        m = (spikes["run"] == r) & (spikes["pop"] == 1) & (spikes["time_ms"] >= t_max - STEADY_MS)
        rate = np.bincount(spikes["neuron_id"][m] - lo, minlength=n)[:n] / (STEADY_MS / 1000)
        return exc, inh, rate

    # --- G: per-neuron excitation vs inhibition ---
    ax = fig.add_subplot(grid[2, 1])
    for r in (ref_s, ref_p):
        if r is None:
            continue
        exc, inh, _ = per_neuron(r)
        ax.scatter(exc, inh, s=8, color=info[r]["color"], alpha=0.6, label=info[r]["label"])
        if info[r]["plastic"] and np.std(exc) > 0:
            cc = np.corrcoef(exc, inh)[0, 1]
            ax.text(0.03, 0.95, f"iSTDP: r = {cc:.2f}", transform=ax.transAxes, fontsize=9,
                    color=info[r]["color"], va="top")
    style(ax, "G  Per E neuron: total excitatory vs. inhibitory weight")
    ax.set_xlabel("summed Input→E weight"); ax.set_ylabel("summed I→E weight")
    ax.set_yscale("symlog", linthresh=1)
    ax.legend(fontsize=8, loc="lower right", markerscale=2)

    # --- H: per-neuron rate distribution ---
    ax = fig.add_subplot(grid[2, 2])
    rates = {r: per_neuron(r)[2] for r in (ref_s, ref_p) if r is not None}
    hi = max(np.percentile(v, 99) for v in rates.values()) if rates else 1
    bins = np.linspace(0, max(hi, 1), 40)
    for r, v in rates.items():
        ax.hist(v, bins=bins, histtype="stepfilled", alpha=0.45, color=info[r]["color"],
                label=f"{info[r]['label']}  (CV {np.std(v) / max(np.mean(v), 1e-9):.2f})")
    style(ax, f"H  Firing rates of individual E neurons (last {STEADY_MS / 1000:.0f} s)")
    ax.set_xlabel("rate [Hz]"); ax.set_ylabel("E neurons")
    ax.legend(fontsize=8)

    # --- I/J: rasters, last 500 ms ---
    window = 500.0
    for k, r in enumerate((ref_s, ref_p)):
        if r is None:
            continue
        ax = fig.add_subplot(grid[3, k])
        m = (spikes["run"] == r) & (spikes["time_ms"] >= t_max - window)
        for pop, c, name in ((1, E_COLOR, "E"), (2, I_COLOR, "I")):
            mm = m & (spikes["pop"] == pop)
            ax.scatter(spikes["time_ms"][mm] / 1000, spikes["neuron_id"][mm], s=1.5, c=c,
                       marker="|", linewidths=1.0, label=name)
        style(ax, f"{'IJ'[k]}  Raster, last {window:.0f} ms ({info[r]['label']})")
        ax.set_xlabel("time [s]"); ax.set_ylabel("neuron id")
        ax.set_ylim(e_ids.min() - 5, (i_ids.max() if i_ids.size else e_ids.max()) + 5)
        ax.legend(loc="upper right", markerscale=6, fontsize=8, framealpha=0.9)

    # --- K: membrane potential of one E neuron ---
    ax = fig.add_subplot(grid[3, 2])
    for r in (ref_s, ref_p):
        if r is None:
            continue
        m = (pot["run"] == r) & (pot["time_ms"] >= t_max - 200)
        ax.plot(pot["time_ms"][m] / 1000, pot["potential"][m], color=info[r]["color"],
                ls=info[r]["ls"], lw=0.9, label=info[r]["label"])
    style(ax, "K  Membrane potential of one E neuron (last 200 ms)")
    ax.set_xlabel("time [s]"); ax.set_ylabel("voltage [mV]")
    ax.legend(fontsize=8)

    out = Path(out_path)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"Saved summary to {out.resolve()}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    return fig


if __name__ == "__main__":
    visualize(*sys.argv[1:2])
