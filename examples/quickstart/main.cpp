/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

// -----------------------------------------------------------------------------
//  A minimal SNN simulation. Unlike the other examples, the whole network
//  is hardcoded in place.
//
//  Architecture (2 layers, feed-forward):
//      Input  layer :  500 LIF neurons that fire spontaneously (~20 Hz)
//      Output layer :  500 LIF neurons, driven purely by the input layer
//      Synapses     :  each input neuron fans out to 200 random output
//                      neurons -> 500 * 200 = 100k excitatory synapses
//      Plasticity   :  every synapse learns via spike-timing-dependent
//                      plasticity (STDP)
//      Total        :  1k neurons, 100k synapses
// -----------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <braincel/Log.h>
#include <braincel/Metrics.h>
#include <braincel/build.h>
#include "spiking/Network.h"

// Path to the (venv) Python interpreter, injected by CMake so matplotlib is found.
#ifndef VENV_PYTHON
#define VENV_PYTHON "python3"
#endif

// ----- Network / simulation -----------------------------
namespace cfg {
    constexpr int   INPUT_NEURONS  = 500;     // layer 1 size
    constexpr int   OUTPUT_NEURONS = 500;     // layer 2 size
    constexpr int   FAN_OUT        = 200;     // synapses per input neuron
    constexpr float SYN_WEIGHT     = 0.30f;   // initial excitatory synaptic weight
    constexpr float INPUT_RATE_HZ  = 20.0f;   // spontaneous firing rate of the input layer
    constexpr int   MONITOR_ID     = 500;     // neuron whose membrane potential we trace

    // LIF
    constexpr float LIF_THRESHOLD  = -50.f;   // [mV] spike when membrane voltage exceeds
    constexpr float LIF_TAU        =  20.f;   // [ms] membrane time constant
    constexpr float LIF_REST       = -70.f;   // [mV] resting / reset potential after a spike
    constexpr float LIF_RESIST     =  10.f;   // [MOhm] membrane resistance
    constexpr float LIF_REVERSE    = -60.f;   // [mV] leak reversal potential Vm decays toward
    constexpr float LIF_ADAPT      =  0.f;    // [mV] threshold-adaptation amount per spike (0 = off)
    constexpr float LIF_ADAPT_TAU  =  100.f;  // [ms] decay time constant of adaptation
    constexpr float LIF_REFRACT    =  2.f;    // [ms] refractory period after a spike

    // STDP plasticity
    constexpr float LEARNING_RATE  = 0.01f;   // step size of each weight update
    constexpr float STDP_TAU_MS    = 20.0f;   // pre/post trace time constant
    constexpr float STDP_A_PLUS    = 1.0f;    // potentiation amplitude (pre-before-post)
    constexpr float STDP_A_MINUS   = 1.0f;    // depression amplitude (post-before-pre)
    constexpr float STDP_MU        = 1.0f;    // weight-dependence: 1 = multiplicative
    constexpr float W_MAX          = 1.0f;    // upper weight limit

    constexpr float TIMESTEP_MS    = 1.0f;    // integration step
    constexpr float DURATION_MS    = 1000.0f; // biological time to simulate
    constexpr unsigned SEED        = 42;      // reproducible wiring

    constexpr bool  USE_CUDA       = true;    // false = run on CPU, true = run membranes on the GPU
}

// Plasticity rule type definition
// -> a Pre-rule lives on each axon
// -> a Post-rule on each soma
// -> a Connection-rule on each synapse
using STDPPreRule  = PreRule<SparseSTDPf, AxonType>;
using STDPPostRule = PostRule<SparseSTDPf, SomaType>;
using STDPConRule  = ConnectionRule<SparseSTDPf, AxonType, SomaType, AxoSomaticSynapseType>;

// Create a single LIF neuron
static NeuronType& addNeuron(Network& net, int id, const std::string& layer, float baselineHz) {
    auto& neuron = net.neurons().emplace<NeuronType>();
    neuron.setId(id);
    neuron.setClassName(layer);      // also names & ids the soma and axon
    neuron.setSimulated(true);

    auto& membrane = neuron.emplaceMembrane<LifMembrane>(cfg::LIF_THRESHOLD, cfg::LIF_TAU,
                                                         cfg::LIF_REST, cfg::LIF_RESIST,
                                                         cfg::LIF_REVERSE, baselineHz,
                                                         cfg::LIF_ADAPT, cfg::LIF_ADAPT_TAU,
                                                         cfg::LIF_REFRACT);
    membrane.setSeed(id);

    auto& axon = neuron.axon();
    axon.setTransmitter(Glutamate);  // excitatory
    axon.setDelay(0.f);
    return neuron;
}

int main(int argc, char* argv[]) {
    // Logging
    Log::configureFromEnv();
    Metrics::configureFromEnv();

    const std::string outPath = argc > 1 ? argv[1] : "quickstart_spikes.csv";
    const std::string potentialPath = argc > 2 ? argv[2] : "quickstart_potential.csv";
    const std::string weightsPath = argc > 3 ? argv[3] : "quickstart__weights.csv";

    // --- BUILD ---
    const auto stb = std::chrono::high_resolution_clock::now();

    Network net;
    auto& neurons = net.neurons();
    neurons.reserve<NeuronType>(cfg::INPUT_NEURONS + cfg::OUTPUT_NEURONS);

    // Keep raw handles
    std::vector<AxonType*> inputAxons;   inputAxons.reserve(cfg::INPUT_NEURONS);
    std::vector<SomaType*> outputSomata;  outputSomata.reserve(cfg::OUTPUT_NEURONS);
    NeuronType* monitored = nullptr;

    // Layer 1
    int id = 0;
    for (int i = 0; i < cfg::INPUT_NEURONS; ++i) {
        auto& n = addNeuron(net, id, "Input", cfg::INPUT_RATE_HZ);
        inputAxons.push_back(&n.axon());
        if (id == cfg::MONITOR_ID) monitored = &n;
        ++id;
    }

    // Layer 2
    for (int i = 0; i < cfg::OUTPUT_NEURONS; ++i) {
        auto& n = addNeuron(net, id, "Output", 0.0f);
        outputSomata.push_back(&n.soma());
        if (id == cfg::MONITOR_ID) monitored = &n;
        ++id;
    }

    neurons.reserveUpdateStorage<NeuronType>(1.0f);
    neurons.reserveActive();
    neurons.clearActive();

    // Connect input -> output
    // Each input axon makes fan-out STDP synapses onto randomly chosen output somata
    std::mt19937 gen(cfg::SEED);
    std::vector<int> targets(cfg::OUTPUT_NEURONS);
    std::iota(targets.begin(), targets.end(), 0);

    long synapses = 0;
    for (auto* axon : inputAxons) {
        std::shuffle(targets.begin(), targets.end(), gen);   // pick distinct targets
        axon->reserveOutputs<AxoSomaticSynapseType>(cfg::FAN_OUT);

        // Per-axon pre-rule
        // Shared by all output synapses of the neuron
        auto& preRule = axon->hasProcess<STDPPreRule>()
            ? axon->process<STDPPreRule>()
            : axon->emplaceProcess<STDPPreRule>(*axon, /*jump*/ 1.0f, cfg::STDP_TAU_MS);

        for (int k = 0; k < cfg::FAN_OUT; ++k) {
            auto* soma = outputSomata[targets[k]];

            auto& syn = axon->emplaceOutput<AxoSomaticSynapseType>(*soma);
            syn.setWeight(cfg::SYN_WEIGHT);
            syn.setDelay(0.0f);
            syn.setLearningRate(cfg::LEARNING_RATE);
            syn.setMinWeight(0.0f);
            syn.setMaxWeight(cfg::W_MAX);
            syn.setWeightClip(true);

            // Per-soma post-rule
            // Whichever axon reaches it first instantiates it
            auto& postRule = soma->hasProcess<STDPPostRule>()
                ? soma->process<STDPPostRule>()
                : soma->emplaceProcess<STDPPostRule>(*soma, /*jump*/ 1.0f, cfg::STDP_TAU_MS);

            // Per-synapse connection-rule
            // Ties pre- and post-rules together
            syn.emplaceProcess<STDPConRule>(syn, *axon, *soma, preRule, postRule,
                                            cfg::STDP_A_PLUS, cfg::STDP_A_MINUS,
                                            cfg::STDP_MU, cfg::STDP_MU, cfg::W_MAX);
            ++synapses;
        }
    }

    // Collect weights
    auto weightStats = [&](double& mean, float& lo, float& hi) {
        double sum = 0; lo = 1e30f; hi = -1e30f; long n = 0;
        for (auto* axon : inputAxons)
            axon->forEachOutput([&](auto& syn) {
                const float w = syn.weight();
                sum += w; lo = std::min(lo, w); hi = std::max(hi, w); ++n;
            });
        mean = n ? sum / n : 0.0;
    };
    double meanW0; float loW0, hiW0;
    weightStats(meanW0, loW0, hiW0);

    const auto etb = std::chrono::high_resolution_clock::now();

    const double builtMs = std::chrono::duration<double, std::milli>(etb - stb).count();

    Log::heading("Brain-CEL Framework Quickstart Example");
    Log::fields({
        {"neurons",  std::format("{} ({} input + {} output)", Log::count(net.numNeurons()), cfg::INPUT_NEURONS, cfg::OUTPUT_NEURONS)},
        {"synapses", Log::count(synapses)},
        {"built",    Log::duration(builtMs)},
    }, 1);

    // --- SIMULATE ---
    net.setTimestep(cfg::TIMESTEP_MS);
    const int totalTicks = static_cast<int>(cfg::DURATION_MS / cfg::TIMESTEP_MS);

    // Device selection
    // callback must be on so Vm is copied back each tick for the potential trace
    if (cfg::USE_CUDA) {
        neurons.setCallbackMembrane(true);
        neurons.setCuda(true);
    }
    std::ofstream csv(outPath);
    csv << "time_ms,neuron_id,layer\n";
    std::ofstream pcsv(potentialPath);
    pcsv << "time_ms,potential\n";

    long spikeCount = 0;
    const auto st = std::chrono::high_resolution_clock::now();

    LiveView& live = Log::live();
    live.setTitle("Simulating");
    live.show();

    // Main loop
    for (int tick = 0; tick < totalTicks; ++tick) {
        net.cycle();   // start a new timestep -> clears per-tick spike states
        net.tick();    // integrate -> fire -> propagate

        // Activity logging
        const float t = tick * cfg::TIMESTEP_MS;
        neurons.forEachEnumerate([&](const size_t i, auto& neuron) {
            if (neuron.spike()) {
                const int layer = static_cast<int>(i) < cfg::INPUT_NEURONS ? 0 : 1;
                csv << t << ',' << neuron.id() << ',' << layer << '\n';
                ++spikeCount;
            }
        });

        // Membrane potential logging
        if (monitored) pcsv << t << ',' << monitored->potential() << '\n';

        live.setProgress(tick + 1, totalTicks);
        live.set("spikes", static_cast<long long>(spikeCount));
        live.refresh();
    }
    live.refresh(true);
    live.hide();

    const auto et = std::chrono::high_resolution_clock::now();
    if (cfg::USE_CUDA) neurons.setCuda(false);   // free GPU resources
    csv.close();
    pcsv.close();

    // --- COUT & VISUALIZATION ---
    const double elMs = std::chrono::duration<double, std::milli>(et - st).count();

    double meanW1; float loW1, hiW1;
    weightStats(meanW1, loW1, hiW1);

    Log::blank();
    Log::section("Simulation");
    Log::fields({
        {"device",    cfg::USE_CUDA ? std::string("CPU-GPU Hybrid (CUDA)") : std::string("CPU")},
        {"simulated", Log::duration(cfg::DURATION_MS)},
        {"elapsed",   Log::duration(elMs)},
        {"RTF",       Log::ratio(elMs / cfg::DURATION_MS) + " x"},
        {"spikes",    std::format("{}  ->  {}", Log::count(spikeCount), outPath)},
        {"STDP",      std::format("mean {:.3f} -> {:.3f}  (range {:.3f} .. {:.3f})", meanW0, meanW1, loW1, hiW1)},
    }, 1);
    Log::blank();

    // Log every initial + final weight
    std::ofstream wcsv(weightsPath);
    wcsv << "w_init,w_final\n";
    for (auto* axon : inputAxons)
        axon->forEachOutput([&](auto& syn) {
            wcsv << cfg::SYN_WEIGHT << ',' << syn.weight() << '\n';
        });
    wcsv.close();

    // Visualize
    const std::string cmd = std::string(VENV_PYTHON) + " visualize.py "
                          + outPath + " " + weightsPath + " " + potentialPath;
    std::system(cmd.c_str());
    return 0;
}
