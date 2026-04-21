# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Marcel Rinder

import nest
import numpy as np
import time
import sys
import argparse

def write(line, logger):
    print(line, end="")
    logger.write(line)
    logger.flush()

REPETITIONS    = 3
SIMTIME        = 10000
FIXED_SYNAPSES = 1000000
FIXED_NEURONS  = 2000
NEURON_COUNTS  = list(range(1000, 11000, 1000))
SYNAPSE_COUNTS_M = [0.04, 0.20, 0.40, 0.80, 1.20, 1.60, 2.00, 2.40, 2.80, 3.20, 3.60, 4.00]
RATES          = [3.0, 5.0, 10.0, 20.0]

NEURON_PARAMS = {
    "C_m":    250.0,
    "tau_m":   10.0,
    "t_ref":    2.0,
    "E_L":    -70.0,
    "V_th":   -55.0,
    "V_reset": -70.0,
}
STDP_PARAMS = {
    "synapse_model": "stdp_synapse",
    "weight":  0.00001,
    "delay":   1.0,
    "Wmax":    0.01,
    "lambda":  0.05,
    "alpha":   1.0,
    "tau_plus": 20.0,
    "mu_plus":  1.0,
    "mu_minus": 1.0,
}
nest.set_verbosity("M_WARNING")


def run_once(n_per_layer, outdegree, rate, nthreads=1):
    nest.ResetKernel()
    nest.SetKernelStatus({"local_num_threads": nthreads, "resolution": 1.0, "rng_seed": 1})

    t0 = time.perf_counter()

    neurons_1 = nest.Create("iaf_psc_delta", n_per_layer, params=NEURON_PARAMS)
    neurons_2 = nest.Create("iaf_psc_delta", n_per_layer, params=NEURON_PARAMS)

    pg1 = nest.Create("poisson_generator", n_per_layer, {"rate": rate})
    pg2 = nest.Create("poisson_generator", n_per_layer, {"rate": rate})
    nest.Connect(pg1, neurons_1, "one_to_one", syn_spec={"weight": 10000})
    nest.Connect(pg2, neurons_2, "one_to_one", syn_spec={"weight": 10000})

    nest.Connect(
        neurons_1, neurons_2,
        {"rule": "fixed_outdegree", "outdegree": outdegree},
        syn_spec=STDP_PARAMS,
    )

    build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    nest.Simulate(SIMTIME)
    sim_s = time.perf_counter() - t1

    return build_s, sim_s

def run_batch(n_per_layer, outdegree, rate, nthreads=1):
    builds, sims = [], []
    for _ in range(REPETITIONS):
        b, s = run_once(n_per_layer, outdegree, rate, nthreads)
        builds.append(b)
        sims.append(s)
    bm, bs = np.mean(builds), np.std(builds)
    sm, ss = np.mean(sims),   np.std(sims)
    rtf     = sm / (SIMTIME / 1000)
    speedup = 1.0 / rtf
    return bm, bs, sm, ss, rtf, speedup


def sweep_neurons(rate, nthreads=1, logger=None):
    label = f"NEST_n_{int(rate)}hz"
    write(f"\nSweep Neurons | rate={rate} Hz | device=CPU | label={label}\n", logger)

    for n in NEURON_COUNTS:
        outdegree = max(1, FIXED_SYNAPSES // n)
        actual_syn = outdegree * n
        syn_m  = actual_syn / 1e6
        density = actual_syn / (n * n)

        bm, bs, sm, ss, rtf, speedup = run_batch(n, outdegree, rate, nthreads)

        write(
            f"syn={syn_m:.2f}M | n1/2={n} | total_n={2*n} | density={density:.4f} | "
            f"build-et={bm:.2f}s ± {bs:.2f} | sim-et={sm:.2f}s ± {ss:.2f} | "
            f"speedup={speedup:.2f} | rtf={rtf:.2f}\n", logger
        )

def sweep_synapses(rate, nthreads=1, logger=None):
    label = f"NEST_s_{int(rate)}hz"
    write(f"\nSweep Synapses | rate={rate} Hz | device=CPU | label={label}\n", logger)

    n = FIXED_NEURONS
    for syn_m_target in SYNAPSE_COUNTS_M:
        target_syn = syn_m_target * 1e6
        outdegree  = max(1, round(target_syn / n))
        actual_syn = outdegree * n
        actual_m   = actual_syn / 1e6

        bm, bs, sm, ss, rtf, speedup = run_batch(n, outdegree, rate, nthreads)

        write(
            f"syn={actual_m:.2f}M | outdegree={outdegree}/neuron | "
            f"build-et={bm:.2f}s ± {bs:.2f} | sim-et={sm:.2f}s ± {ss:.2f} | "
            f"speedup={speedup:.2f} | rtf={rtf:.2f}\n", logger
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEST benchmark")
    parser.add_argument("-o", "--output", default="nest_bench.txt", help="output file")
    parser.add_argument("-t", "--threads", type=int, default=1, help="number of NEST threads")
    args = parser.parse_args()
    
    OUT_FILE = args.output
    log = open(OUT_FILE, "w")

    for rate in RATES:
        sweep_neurons(rate, args.threads, log)
    for rate in RATES:
        sweep_synapses(rate, args.threads, log)

    log.close()
    print(f"\n[nest_bench] All sweeps complete.")