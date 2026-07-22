/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <braincel/Log.h>
#include <braincel/Metrics.h>
#include <braincel/Simulator.h>
#include <braincel/config.h>


static int nodeIndex(const json& buildParams, const std::string& name) {
    const auto& nodes = buildParams["scene"]["nodes"];
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
        if (nodes[i].value("name", "") == name) return i;
    throw std::runtime_error("Node not found: " + name);
}

struct RunResult {
    double constructionTime = 0;
    double simulationTime   = 0;
    double rtf              = 0;
};
struct Stats  { double mean, std; };
struct Result { Stats build, sim, rtf; };
static Stats stat(const std::vector<double>& v) {
    const double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0;
    for (const double x : v) var += (x - m) * (x - m);
    return { m, std::sqrt(var / v.size()) };
}

static void printRow(const double numSyn, const int outdegree, const int total,
                     const double density, const double speedup,
                     const Result& r, std::ofstream& log) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "syn=" << numSyn << "M | "
        << "n1/2=" << outdegree << " | "
        << "total_n=" << total << " | "
        << "density=" << density << " | "
        << "build-et=" << r.build.mean << "s ± " << r.build.std << " | "
        << "sim-et="   << r.sim.mean   << "s ± " << r.sim.std   << " | "
        << "speedup="  << speedup       << " | "
        << "rtf="      << r.rtf.mean   << "\n";
    std::cout << oss.str();
    log       << oss.str();
    log.flush();
}

static RunResult run(const json& buildParams, const json& simParams) {
    Simulator sim;

    const auto t0 = std::chrono::high_resolution_clock::now();
    sim.build(buildParams);
    const auto t1 = std::chrono::high_resolution_clock::now();
    sim.simulate(simParams);
    const auto t2 = std::chrono::high_resolution_clock::now();

    const double bioMs = simParams.value("duration", 10000.0);

    RunResult r;
    r.constructionTime = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.simulationTime   = std::chrono::duration<double, std::milli>(t2 - t1).count();
    r.rtf              = r.simulationTime / bioMs;
    return r;
}
static Result runBatch(const json& buildParams, const json& simParams, const int repetitions) {
    std::vector<double> builds, sims, rtfs;
    for (int r = 0; r < repetitions; ++r) {
        auto [constructionTime, simulationTime, rtf] = run(buildParams, simParams);
        builds.push_back(constructionTime / 1000.0);
        sims  .push_back(simulationTime   / 1000.0);
        rtfs  .push_back(rtf);
    }
    return { stat(builds), stat(sims), stat(rtfs) };
}

static void sweepNeurons(const json& buildParams, const json& simParams,
                         const std::vector<int>&    neuronCounts,
                         const long long                  numSynapsesFixed,
                         double                     firingRate,
                         const std::string&         device,
                         const int                        repetitions,
                         const std::string&         label,
                         std::ofstream&             log) {
    std::ostringstream header;
    header << "\nSweep Neurons | rate=" << firingRate
           << " Hz | device=" << device << " | label=" << label << "\n";
    std::cout << header.str();
    log       << header.str();
    log.flush();

    const int iL1  = nodeIndex(buildParams, "Layer 1");
    const int iL2  = nodeIndex(buildParams, "Layer 2");
    const int iCon = 0;

    for (int n : neuronCounts) {
        json curBuildParams = buildParams;
        json curSimParams = simParams;

        // Set neuron counts
        setField(curBuildParams, "scene.nodes." + std::to_string(iL1) + ".node.number", n);
        setField(curBuildParams, "scene.nodes." + std::to_string(iL2) + ".node.number", n);

        // Set firing rates
        setField(curBuildParams, "scene.nodes." + std::to_string(iL1) + ".membrane.baseline_fire", firingRate);
        setField(curBuildParams, "scene.nodes." + std::to_string(iL2) + ".membrane.baseline_fire", firingRate);

        // Set device
        setField(curBuildParams, "scene.nodes." + std::to_string(iL1) + ".membrane.device", device);
        setField(curBuildParams, "scene.nodes." + std::to_string(iL2) + ".membrane.device", device);
        setField(curSimParams, "device", device);

        const double density = static_cast<double>(numSynapsesFixed) / (static_cast<double>(n) * static_cast<double>(n));
        setField(curBuildParams, "scene.connections." + std::to_string(iCon) + ".params.connection.density", density);

        const double actualSyn = density * n * n;
        const double numSyn = actualSyn / 1e6;

        auto res = runBatch(curBuildParams, curSimParams, repetitions);

        const double speedup = 1.0 / res.rtf.mean;
        printRow(numSyn, n, 2 * n, density, speedup, res, log);
    }
}

static void sweepSynapses(const json& buildParams, const json& simParams,
                          int                        numNeuronsFixed,
                          const std::vector<double>& synapseCounts,
                          double                     firingRate,
                          const std::string&         device,
                          int                        repetitions,
                          const std::string&         label,
                          std::ofstream&             log) {
    std::ostringstream header;
    header << "\nSweep Synapses | rate=" << firingRate
           << " Hz | device=" << device << " | label=" << label << "\n";
    std::cout << header.str();
    log       << header.str();
    log.flush();

    const int iL1  = nodeIndex(buildParams, "Layer 1");
    const int iL2  = nodeIndex(buildParams, "Layer 2");
    const int iCon = 0;

    json bpFixed = buildParams;
    setField(bpFixed, "scene.nodes." + std::to_string(iL1) + ".node.number", numNeuronsFixed);
    setField(bpFixed, "scene.nodes." + std::to_string(iL2) + ".node.number", numNeuronsFixed);
    setField(bpFixed, "scene.nodes." + std::to_string(iL1) + ".membrane.baseline_fire", firingRate);
    setField(bpFixed, "scene.nodes." + std::to_string(iL2) + ".membrane.baseline_fire", firingRate);
    setField(bpFixed, "scene.nodes." + std::to_string(iL1) + ".membrane.device", device);
    setField(bpFixed, "scene.nodes." + std::to_string(iL2) + ".membrane.device", device);

    for (double numSyn : synapseCounts) {
        json cur = bpFixed;

        const double targetSyn = numSyn * 1e6;
        const double density   = targetSyn / (static_cast<double>(numNeuronsFixed) * static_cast<double>(numNeuronsFixed));
        const int    outdegree = std::max(1, static_cast<int>(std::round(density * numNeuronsFixed)));

        setField(cur, "scene.connections." + std::to_string(iCon) + ".params.connection.density", density);

        const double actualSyn = static_cast<double>(outdegree) * numNeuronsFixed;
        const double actualM   = actualSyn / 1e6;

        auto [build, sim, rtf] = runBatch(cur, simParams, repetitions);

        const double speedup = 1.0 / rtf.mean;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << "syn=" << actualM << "M | "
            << "outdegree=" << outdegree << "/neuron | "
            << "build-et=" << build.mean << "s ± " << build.std << " | "
            << "sim-et="   << sim.mean   << "s ± " << sim.std   << " | "
            << "speedup="  << speedup         << " | "
            << "rtf="      << rtf.mean   << "\n";
        std::cout << oss.str();
        log       << oss.str();
        log.flush();
    }
}


int main(int argc, char* argv[]) {
    // Logging
    Log::configureFromEnv();
    Metrics::configureFromEnv();

    const std::string buildParamsPath = argc > 1 ? argv[1] : "buildparams.json";
    const std::string simParamsPath   = argc > 2 ? argv[2] : "simparams.json";
    const std::string logPath         = argc > 3 ? argv[3] : "braincel_bench.txt";

    const int REPETITIONS    = argc > 4 ? std::atoi(argv[4]) : 3;
    const int FIXED_SYNAPSES = argc > 5 ? std::atoi(argv[5]) : 1000000;
    const int FIXED_NEURONS  = argc > 6 ? std::atoi(argv[6]) : 2000;

    const std::vector NEURON_COUNTS  = { 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };
    const std::vector SYNAPSE_COUNTS = { 0.04, 0.20, 0.40, 0.80, 1.20, 1.60, 2.00, 2.40, 2.80, 3.20, 3.60, 4.00 };
    const std::vector RATES          = { 3.0, 5.0, 10.0, 20.0 };
    const std::vector<std::string> DEVICES = { "GPU Compute", "CPU" };

    // Load configs
    const auto buildParams = loadJson(buildParamsPath);
    const auto simParams   = loadJson(simParamsPath);
    std::ofstream log(logPath, std::ios::out | std::ios::trunc);

    // Run sweep
    for (const auto& device : DEVICES) {
        const std::string devTag = (device == "CPU") ? "CPU" : "GPU";
        for (const double rate : RATES) {
            sweepNeurons(buildParams, simParams,
                         NEURON_COUNTS, FIXED_SYNAPSES,
                         rate, device, REPETITIONS,
                         devTag + "_n_" + std::to_string(static_cast<int>(rate)) + "hz",
                         log);
        }
        for (const double rate : RATES) {
            sweepSynapses(buildParams, simParams,
                          FIXED_NEURONS, SYNAPSE_COUNTS,
                          rate, device, REPETITIONS,
                          devTag + "_s_" + std::to_string(static_cast<int>(rate)) + "hz",
                          log);
        }
    }

    std::cout << "\nAll sweeps complete. Results in: " << logPath << "\n";
    return 0;
}
