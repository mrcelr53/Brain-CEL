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

static void sweepRates(const json& buildParams, const json& simParams,
                       const bool                 forwardTriggered,
                       const std::vector<double>& firingRates,
                       int                        repetitions,
                       const std::string&         label,
                       std::ofstream&             log) {
    std::ostringstream header;
    header << "\nSweep Firing Rates | "
           << (forwardTriggered ? "FT-STDP" : "Bi-STDP")
           << " | label=" << label << "\n";
    std::cout << header.str();
    log       << header.str();
    log.flush();


    const int iL1  = nodeIndex(buildParams, "Layer 1");
    const int iL2  = nodeIndex(buildParams, "Layer 2");
    const int iCon = 0;

    json curBuildParams = buildParams;
    setField(curBuildParams, "scene.connections." + std::to_string(iCon) + ".params.learning.forward_triggered", forwardTriggered);

    for (double preFiringRate : firingRates) {
        for (double postFiringRate : firingRates) {
            json cur = curBuildParams;

            setField(cur, "scene.nodes." + std::to_string(iL1) + ".membrane.baseline_fire", preFiringRate);
            setField(cur, "scene.nodes." + std::to_string(iL2) + ".membrane.baseline_fire", postFiringRate);

            auto [build, sim, rtf] = runBatch(cur, simParams, repetitions);

            const double speedup = 1.0 / rtf.mean;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2)
                << "pre-rate=" << preFiringRate << "Hz | "
                << "post-rate=" << postFiringRate << "Hz | "
                << "build-et=" << build.mean << "s ± " << build.std << " | "
                << "sim-et="   << sim.mean   << "s ± " << sim.std   << " | "
                << "speedup="  << speedup         << " | "
                << "rtf="      << rtf.mean   << "\n";
            std::cout << oss.str();
            log       << oss.str();
            log.flush();
        }
    }
}

int main(int argc, char* argv[]) {
    const std::string buildParamsPath = argc > 1 ? argv[1] : "buildparams.json";
    const std::string simParamsPath   = argc > 2 ? argv[2] : "simparams.json";
    const std::string logPath         = argc > 3 ? argv[3] : "braincel_bench.txt";

    constexpr int REPETITIONS    = 3;
    const std::vector RATES = {1., 3., 5., 10., 15., 20., 30., 40., 50.};

    // Load configs
    const auto buildParams = loadJson(buildParamsPath);
    const auto simParams   = loadJson(simParamsPath);
    std::ofstream log(logPath, std::ios::out | std::ios::trunc);

    // Run sweep
    sweepRates(buildParams, simParams, false, RATES, REPETITIONS, "rates_bitrig", log);
    sweepRates(buildParams, simParams, true, RATES, REPETITIONS, "rates_fwtrig", log);

    std::cout << "\nAll sweeps complete. Results in: " << logPath << "\n";
    return 0;
}
