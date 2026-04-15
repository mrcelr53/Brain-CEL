/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
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

#include <braincel/build.h>
#include <braincel/config.h>
#include "spiking/Network.h"


struct TimingResult {
    std::vector<double> dt;
    std::vector<double> dw;
};

static TimingResult runTimingProtocol(const json& buildParams, const json& simParams) {
    const double timestep     = simParams.value("time_step", 1.0);
    const int    resolution   = simParams.value("resolution", 100);
    const int    repetitions  = simParams.value("repetitions", 100);
    const double timingRange  = simParams.value("timing_range", 50.0);
    const double frequency    = simParams.value("frequency", 10.0);
    const double clampVoltage = simParams.value("clamp_voltage", -65.0);

    const int numNeurons = resolution * 2;

    Network net;

    auto& neurons = net.neurons();
    neurons.reserve<NeuronType>(numNeurons);

    auto* preView  = net.addNeuronView("Pre");
    auto* postView = net.addNeuronView("Post");

    std::vector<Timed*> preMap, postMap;

    for (int i = 0; i < numNeurons; ++i) {
        auto& neuron = neurons.emplace<NeuronType>();
        neuron.setClassUuid(numNeurons);
        neuron.setClassName("Dummy Neuron");
        neuron.setId(i);
        neuron.setClampedPoissonRate(0.);
        neuron.setClampedPotential(clampVoltage);

        auto& soma = neuron.soma();
        soma.setId(i);

        auto& axon = neuron.axon();
        axon.setId(i);
        axon.setTransmitter(Glutamate);
        axon.setDelay(0.);

        if (i < numNeurons / 2) {
            preMap.push_back(&axon);
            preView->insert(&neuron);
        } else {
            postMap.push_back(&soma);
            postView->insert(&neuron);
        }
    }

    neurons.reserveUpdateStorage<NeuronType>(1.0f);
    neurons.reserveActive();
    neurons.clearActive();

    // Build synapses
    const auto& sceneParams = buildParams.value("scene", json::object());
    const auto  conParams   = sceneParams.value("connections", json::array());

    double initialWeight = 0.5;

    for (const auto& conArg : conParams) {
        const auto conParam      = conArg.value("params", json::object());
        const auto mainParam     = conParam.value("connection", json::object());
        const auto learnParam    = conParam.value("learning", json::object());
        const auto preTermParam  = conParam.value("pre_terminal", json::object());
        const auto postTermParam = conParam.value("post_terminal", json::object());

        const double weight       = mainParam.value("weight", 1.0);
        const double delay        = mainParam.value("delay", 0.0);
        const double learningRate = learnParam.value("rate", 0.0);
        const double spikeDur     = learnParam.value("spike_duration", 0.01);
        const std::string rule    = learnParam.value("rule", std::string("STDP"));
        const bool fwdTriggered   = learnParam.value("forward_triggered", true);
        const double maxWeight    = learnParam.value("max_weight", 10.0);

        initialWeight = weight;

        for (int ci = 0; ci < static_cast<int>(preMap.size()); ++ci) {
            auto* preAxon  = dynamic_cast<AxonType*>(preMap[ci]);
            auto* postSoma = dynamic_cast<SomaType*>(postMap[ci]);
            if (!preAxon || !postSoma) continue;

            preAxon->reserve<AxoSomaticSynapseType>(1);

            auto& syn = preAxon->emplaceOutput<AxoSomaticSynapseType>(*postSoma);
            syn.setId(ci);
            syn.setWeight(initialWeight);
            syn.setDelay(delay);
            syn.setLearningRate(learningRate);
            syn.setWeightClip(false);

            if (rule == "STDP" && fwdTriggered) {
                const double a_plus     = learnParam.value("stdp_a_plus", 1.0);
                const double a_minus    = learnParam.value("stdp_a_minus", 1.0);
                const double mu_plus    = learnParam.value("stdp_mu_plus", 0.0);
                const double mu_minus   = learnParam.value("stdp_mu_minus", 0.0);
                const double pre_jump   = preTermParam.value("stdp_jump_amp", 1.0);
                const double pre_tau    = preTermParam.value("stdp_tau", 20.0);
                const double post_jump  = postTermParam.value("stdp_jump_amp", 1.0);
                const double post_tau   = postTermParam.value("stdp_tau", 20.0);

                using PreRuleType  = PreRule<SparseSTDPf, AxonType>;
                using PostRuleType = PostRule<SparseSTDPf, SomaType>;
                using ConRuleType  = ConnectionRule<SparseSTDPf, AxonType, SomaType, AxoSomaticSynapseType>;

                auto& preRule = preAxon->hasProcess<PreRuleType>()
                    ? preAxon->process<PreRuleType>()
                    : preAxon->emplaceProcess<PreRuleType>(*preAxon, pre_jump, pre_tau);
                auto& postRule = postSoma->hasProcess<PostRuleType>()
                    ? postSoma->process<PostRuleType>()
                    : postSoma->emplaceProcess<PostRuleType>(*postSoma, post_jump, post_tau);
                if (!syn.hasProcess<ConRuleType>())
                    syn.emplaceProcess<ConRuleType>(syn, *preAxon, *postSoma, preRule, postRule,
                                                    a_plus, a_minus, mu_plus, mu_minus, maxWeight);
            }
            else if (rule == "STDP") {
                const double a_plus     = learnParam.value("stdp_a_plus", 1.0);
                const double a_minus    = learnParam.value("stdp_a_minus", 1.0);
                const double mu_plus    = learnParam.value("stdp_mu_plus", 0.0);
                const double mu_minus   = learnParam.value("stdp_mu_minus", 0.0);
                const double pre_jump   = preTermParam.value("stdp_jump_amp", 1.0);
                const double pre_tau    = preTermParam.value("stdp_tau", 20.0);
                const double post_jump  = postTermParam.value("stdp_jump_amp", 1.0);
                const double post_tau   = postTermParam.value("stdp_tau", 20.0);

                using PreRuleType  = PreRule<SparseSTDP, AxonType>;
                using PostRuleType = PostRule<SparseSTDP, SomaType>;
                using ConRuleType  = ConnectionRule<SparseSTDP, AxonType, SomaType, AxoSomaticSynapseType>;

                auto& preRule = preAxon->hasProcess<PreRuleType>()
                    ? preAxon->process<PreRuleType>()
                    : preAxon->emplaceProcess<PreRuleType>(*preAxon, pre_jump, pre_tau);
                auto& postRule = postSoma->hasProcess<PostRuleType>()
                    ? postSoma->process<PostRuleType>()
                    : postSoma->emplaceProcess<PostRuleType>(*postSoma, post_jump, post_tau);
                if (!syn.hasProcess<ConRuleType>())
                    syn.emplaceProcess<ConRuleType>(syn, *preAxon, *postSoma, preRule, postRule,
                                                    a_plus, a_minus, mu_plus, mu_minus);
            }
        }
    }

    // Generate spike trains
    auto timingDT = linspace(-timingRange, timingRange, resolution);

    const auto preSpikes = generateRegularSpiketrain(frequency, timestep, repetitions);
    const int totalTicks = static_cast<int>(preSpikes.size());
    const double duration = totalTicks * timestep;

    std::vector<std::vector<double>> postSpikesList;
    for (double dt : timingDT) {
        postSpikesList.push_back(generateRegularSpiketrain(frequency, timestep, duration, dt));
    }

    // Simulate
    net.setTimestep(static_cast<float>(timestep));

    for (int tick = 0; tick < totalTicks; ++tick) {
        net.cycle();

        neurons.forEachEnumerate([&](const size_t i, auto& neuron) {
            if (i < numNeurons / 2) {
                if (preSpikes.at(tick) != 0)
                    neurons.setActive(i);
            } else {
                const size_t pairIdx = i - numNeurons / 2;
                const auto& postSpikes = postSpikesList.at(pairIdx);
                if (tick < static_cast<int>(postSpikes.size()) && postSpikes.at(tick) != 0)
                    neurons.setActive(i);
            }
        });

        net.tick();
    }

    // Collect weight changes
    std::vector<double> timingDW;
    neurons.forEach([&](auto& neuron) {
        neuron.axon().forEachOutput([&](auto& synapse) {
            timingDW.push_back(synapse.weight() - initialWeight);
        });
    });

    return { timingDT, timingDW };
}

// output
static void writeTimingCSV(const std::string& path,
                           const std::vector<double>& dt,
                           const std::vector<double>& dw) {
    std::ofstream f(path);
    f << "dt_ms,dw\n";
    const int n = std::min(dt.size(), dw.size());
    for (int i = 0; i < n; ++i)
        f << std::fixed << std::setprecision(4) << dt[i] << "," << dw[i] << "\n";
}

int main(int argc, char* argv[]) {
    const std::string buildPath = argc > 1 ? argv[1] : "buildparams.json";
    const std::string simPath   = argc > 2 ? argv[2] : "simparams.json";
    const std::string outPath   = argc > 3 ? argv[3] : "stdp_timing.csv";

    const auto buildParams = loadJson(buildPath);
    const auto simParams   = loadJson(simPath);

    const auto t0 = std::chrono::high_resolution_clock::now();
    auto [dt, dw] = runTimingProtocol(buildParams, simParams);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Timing protocol finished in " << std::fixed
              << std::setprecision(2) << elapsed << " s\n";
    writeTimingCSV(outPath, dt, dw);

    std::cout << "All sweeps complete. Results in: " << outPath << "\n";

    return 0;
}
