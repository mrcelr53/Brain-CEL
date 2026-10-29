/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <braincel/Log.h>
#include <braincel/Metrics.h>
#include <braincel/build.h>
#include "spiking/Network.h"

// Path to the (venv) Python interpreter, injected by CMake so matplotlib is found.
#ifndef VENV_PYTHON
#define VENV_PYTHON "python3"
#endif

// Plasticity rule type definitions (see quickstart)
// Excitatory: forward-triggered STDP
using STDPPreRule   = PreRule<SparseSTDPf, AxonType>;
using STDPPostRule  = PostRule<SparseSTDPf, SomaType>;
using STDPConRule   = ConnectionRule<SparseSTDPf, AxonType, SomaType, AxoSomaticSynapseType>;
// Inhibitory: forward-triggered iSTDP
using ISTDPPreRule  = PreRule<SparseSTDPfi, AxonType>;
using ISTDPPostRule = PostRule<SparseSTDPfi, SomaType>;
using ISTDPConRule  = ConnectionRule<SparseSTDPfi, AxonType, SomaType, AxoSomaticSynapseType>;

enum Pop { INPUT = 0, EXC = 1, INH = 2 };

static json loadJson(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    return json::parse(f);
}

// Create a single LIF neuron
static NeuronType& addNeuron(Network& net, int id, const std::string& pop, const json& lif,
                             float baselineHz, Transmitter tx) {
    auto& neuron = net.neurons().emplace<NeuronType>();
    neuron.setId(id);
    neuron.setClassName(pop);
    neuron.setSimulated(true);

    auto& membrane = neuron.emplaceMembrane<LifMembrane>(lif.value("threshold", -50.f), lif.value("tau", 20.f),
                                                         lif.value("rest", -70.f),      lif.value("resist", 10.f),
                                                         lif.value("reverse", -60.f),   baselineHz,
                                                         lif.value("adapt", 0.f),       lif.value("adapt_tau", 100.f),
                                                         lif.value("refract", 2.f));
    membrane.setSeed(id);

    auto& axon = neuron.axon();
    axon.setTransmitter(tx);   // Glutamate = excitatory, GABA = inhibitory
    axon.setDelay(0.f);
    return neuron;
}

// Synapses from every axon onto fanOut random targets. `attach` adds the
// plasticity rule to each synapse (or nothing, for static synapses).
template <typename Attach>
static long connect(const std::vector<AxonType*>& src, const std::vector<SomaType*>& dst,
                    int fanOut, float weight, float rate, std::mt19937& gen, Attach&& attach) {
    std::vector<int> targets(dst.size());
    std::iota(targets.begin(), targets.end(), 0);
    fanOut = std::min<int>(fanOut, dst.size());
    long n = 0;
    for (auto* axon : src) {
        std::shuffle(targets.begin(), targets.end(), gen);
        axon->reserveOutputs<AxoSomaticSynapseType>(fanOut);
        for (int k = 0; k < fanOut; ++k) {
            auto* soma = dst[targets[k]];
            auto& syn = axon->emplaceOutput<AxoSomaticSynapseType>(*soma);
            syn.setWeight(weight);
            syn.setDelay(0.0f);
            syn.setLearningRate(rate);   // 0 and no rule attached -> static
            syn.setMinWeight(0.0f);
            syn.setWeightClip(true);
            attach(*axon, *soma, syn);
            ++n;
        }
    }
    return n;
}

// Excitatory STDP, identical to the quickstart wiring
static auto stdpAttach(const json& stdp) {
    const float tau = stdp.value("tau", 20.f);
    return [=](AxonType& axon, SomaType& soma, AxoSomaticSynapseType& syn) {
        auto& preRule = axon.hasProcess<STDPPreRule>()
            ? axon.process<STDPPreRule>()
            : axon.emplaceProcess<STDPPreRule>(axon, /*jump*/ 1.0f, tau);
        auto& postRule = soma.hasProcess<STDPPostRule>()
            ? soma.process<STDPPostRule>()
            : soma.emplaceProcess<STDPPostRule>(soma, /*jump*/ 1.0f, tau);
        syn.setMaxWeight(stdp.value("w_max", 1.f));
        syn.emplaceProcess<STDPConRule>(syn, axon, soma, preRule, postRule,
                                        stdp.value("a_plus", 1.f), stdp.value("a_minus", 1.f),
                                        stdp.value("mu", 0.f), stdp.value("mu", 0.f),
                                        stdp.value("w_max", 1.f));
    };
}

// Inhibitory STDP. Its parameters live in the rule states, so they are
// written into the states after construction.
// SparseSTDPfi adds a_plus * pre-trace on each I spike and subtracts
// a_minus * post-trace on each E spike. For homeostasis, E activity must
// strengthen inhibition, hence both amplitudes are negative in
// buildparams.json. A short tau_pre makes the I-spike term a constant
// depression per spike, a long tau_post makes the E-spike term grow with
// the E rate, so the weight settles where both balance.
static auto istdpAttach(const json& istdp) {
    const float tauPre  = istdp.value("tau_pre", 20.f);
    const float tauPost = istdp.value("tau_post", 20.f);
    const float tauElig = istdp.value("tau_eligibility", 40.f);
    const float aPlus   = istdp.value("a_plus", 1.f);
    const float aMinus  = istdp.value("a_minus", 1.f);

    return [=](AxonType& axon, SomaType& soma, AxoSomaticSynapseType& syn) {
        if (!axon.hasProcess<ISTDPPreRule>()) {
            auto& st = axon.emplaceProcess<ISTDPPreRule>(axon).state();
            st.tau_pre = tauPre;
            st.inv_tau_pre = 1.f / tauPre;
        }
        if (!soma.hasProcess<ISTDPPostRule>()) {
            auto& st = soma.emplaceProcess<ISTDPPostRule>(soma).state();
            st.tau_post = tauPost;
            st.inv_tau_post = 1.f / tauPost;
        }
        auto& st = syn.emplaceProcess<ISTDPConRule>(syn, axon, soma, axon.process<ISTDPPreRule>(),
                                                    soma.process<ISTDPPostRule>()).state();
        st.a_plus = aPlus;
        st.a_minus = aMinus;
        st.tau_eligibility = tauElig;
        st.inv_tau_eligibility = 1.f / tauElig;
    };
}

struct Outputs {
    std::ofstream runs, spikes, series, potential, weights;
};

// Build and simulate the network for one entry of "runs"
static void runOnce(int runId, const json& run, const json& build, const json& sim, Outputs& out) {
    const auto& pops = build.at("populations");
    const auto& lif  = build.at("lif");
    const auto& con  = build.at("connections");

    const int   nIn     = pops.at("input").value("size", 500);
    const float inHz    = pops.at("input").value("rate_hz", 20.f);
    const int   nE      = pops.at("excitatory").value("size", 400);
    const int   nI      = pops.at("inhibitory").value("size", 100);
    const float dt      = sim.value("time_step", 1.f);
    const float durMs   = sim.value("duration", 60000.f);
    const float binMs   = sim.value("bin_ms", 100.f);
    const bool  cuda    = sim.value("use_cuda", false);
    const float logFrom = durMs - sim.value("spike_log_ms", durMs);   // only log the final window
    const float gInh    = run.value("g_inh", 1.f);
    const bool  plastic = run.value("plastic", false);
    const std::string label = run.value("label", std::format("run {}", runId));

    // --- BUILD ---
    Network net;
    auto& neurons = net.neurons();
    neurons.reserve<NeuronType>(nIn + nE + nI);

    std::vector<AxonType*> inAxons, eAxons, iAxons;
    std::vector<SomaType*> eSomata, iSomata;
    int id = 0;
    for (int i = 0; i < nIn; ++i, ++id)
        inAxons.push_back(&addNeuron(net, id, "Input", lif, inHz, Glutamate).axon());
    NeuronType* monitored = nullptr;   // first E neuron
    for (int i = 0; i < nE; ++i, ++id) {
        auto& n = addNeuron(net, id, "Excitatory", lif, 0.f, Glutamate);
        if (!monitored) monitored = &n;
        eAxons.push_back(&n.axon());
        eSomata.push_back(&n.soma());
    }
    for (int i = 0; i < nI; ++i, ++id) {
        auto& n = addNeuron(net, id, "Inhibitory", lif, 0.f, GABA);
        iAxons.push_back(&n.axon());
        iSomata.push_back(&n.soma());
    }

    neurons.reserveUpdateStorage<NeuronType>(1.0f);
    neurons.reserveActive();
    neurons.clearActive();

    std::mt19937 gen(sim.value("seed", 42u));
    const auto& inE = con.at("input_to_E");
    const auto& eI  = con.at("E_to_I");
    const auto& iE  = con.at("I_to_E");
    const auto noRule = [](AxonType&, SomaType&, AxoSomaticSynapseType&) {};
    long synapses = 0;
    synapses += connect(inAxons, eSomata, inE.value("fan_out", 160), inE.value("weight", 0.25f),
                        inE.at("stdp").value("learning_rate", 0.01f), gen, stdpAttach(inE.at("stdp")));
    synapses += connect(eAxons, iSomata, eI.value("fan_out", 50), eI.value("weight", 0.7f), 0.f, gen, noRule);
    if (plastic)
        synapses += connect(iAxons, eSomata, iE.value("fan_out", 80), gInh,
                            iE.at("istdp").value("learning_rate", 0.001f), gen, istdpAttach(iE.at("istdp")));
    else
        synapses += connect(iAxons, eSomata, iE.value("fan_out", 80), gInh, 0.f, gen, noRule);

    auto meanWeight = [](const std::vector<AxonType*>& axons) {
        double sum = 0; long n = 0;
        for (auto* axon : axons)
            axon->forEachOutput([&](auto& syn) { sum += syn.weight(); ++n; });
        return n ? sum / n : 0.0;
    };

    out.runs << runId << ',' << label << ',' << gInh << ',' << plastic << '\n';

    // --- SIMULATE ---
    net.setTimestep(dt);
    const int totalTicks = static_cast<int>(durMs / dt);
    const int binTicks   = std::max(1, static_cast<int>(binMs / dt));
    if (cuda) {
        neurons.setCallbackMembrane(true);
        neurons.setCuda(true);
    }

    LiveView& live = Log::live();
    live.setTitle(std::format("Simulating  {}", label));
    live.show();

    long spk[3] = {0, 0, 0};             // spikes per population in current bin
    double excDrive = 0, inhDrive = 0;   // summed input onto E population in current bin
    double rateE = 0, rateI = 0;         // rates of the last bin
    const auto st = std::chrono::high_resolution_clock::now();

    for (int tick = 0; tick < totalTicks; ++tick) {
        net.cycle();
        net.tick();

        const float t = tick * dt;
        neurons.forEachEnumerate([&](const size_t i, auto& neuron) {
            if (!neuron.spike()) return;
            const int idx = static_cast<int>(i);
            const int pop = idx < nIn ? INPUT : idx < nIn + nE ? EXC : INH;
            ++spk[pop];
            if (pop == INPUT) {
                inAxons[idx]->forEachOutput([&](auto& syn) { excDrive += syn.weight(); });
            } else {
                if (t >= logFrom) out.spikes << runId << ',' << t << ',' << neuron.id() << ',' << pop << '\n';
                if (pop == INH) iAxons[idx - nIn - nE]->forEachOutput([&](auto& syn) { inhDrive += syn.weight(); });
            }
        });
        out.potential << runId << ',' << t << ',' << monitored->potential() << '\n';

        if ((tick + 1) % binTicks == 0) {
            const double binS = binTicks * dt * 1e-3;
            rateE = spk[EXC] / binS / nE;
            rateI = spk[INH] / binS / nI;
            out.series << runId << ',' << (tick + 1 - binTicks / 2.0) * dt << ','
                       << spk[INPUT] / binS / nIn << ',' << rateE << ',' << rateI << ','
                       << meanWeight(inAxons) << ',' << meanWeight(iAxons) << ','
                       << excDrive / binS / nE << ',' << inhDrive / binS / nE << '\n';
            for (auto& s : spk) s = 0;
            excDrive = inhDrive = 0;
        }

        live.setProgress(tick + 1, totalTicks);
        live.set("E rate [Hz]", std::format("{:.1f}", rateE));
        live.refresh();
    }
    live.refresh(true);
    live.hide();
    if (cuda) neurons.setCuda(false);

    const double elMs = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - st).count();

    for (auto* axon : inAxons)
        axon->forEachOutput([&](auto& syn) { out.weights << runId << ",0," << syn.post().id() << ',' << syn.weight() << '\n'; });
    for (auto* axon : iAxons)
        axon->forEachOutput([&](auto& syn) { out.weights << runId << ",1," << syn.post().id() << ',' << syn.weight() << '\n'; });

    Log::fields({
        {"run",      label},
        {"synapses", Log::count(synapses)},
        {"rates",    std::format("E {:.1f} Hz | I {:.1f} Hz  (last bin)", rateE, rateI)},
        {"weights",  std::format("Input->E {:.3f} | I->E {:.3f}", meanWeight(inAxons), meanWeight(iAxons))},
        {"elapsed",  Log::duration(elMs)},
    }, 1);
    Log::blank();
}

int main(int argc, char* argv[]) {
    Log::configureFromEnv();
    Metrics::configureFromEnv();

    const json build = loadJson(argc > 1 ? argv[1] : "buildparams.json");
    const json sim   = loadJson(argc > 2 ? argv[2] : "simparams.json");
    const json runs  = sim.value("runs", json::array());

    Log::heading("Brain-CEL Homeostasis Example");
    Log::fields({
        {"device",   sim.value("use_cuda", false) ? std::string("CPU-GPU Hybrid (CUDA)") : std::string("CPU")},
        {"simulate", Log::duration(sim.value("duration", 60000.0))},
        {"runs",     std::format("{}", runs.size())},
    }, 1);
    Log::blank();

    Outputs out;
    out.runs.open("homeostasis_runs.csv");
    out.runs << "run,label,g_inh,plastic\n";
    out.spikes.open("homeostasis_spikes.csv");
    out.spikes << "run,time_ms,neuron_id,pop\n";
    out.series.open("homeostasis_series.csv");
    out.series << "run,time_ms,rate_in,rate_E,rate_I,w_exc,w_inh,exc_drive,inh_drive\n";
    out.potential.open("homeostasis_potential.csv");
    out.potential << "run,time_ms,potential\n";
    out.weights.open("homeostasis_weights.csv");
    out.weights << "run,inhibitory,target,w_final\n";

    for (size_t r = 0; r < runs.size(); ++r) runOnce(static_cast<int>(r), runs[r], build, sim, out);

    out.runs.close(); out.spikes.close(); out.series.close(); out.potential.close(); out.weights.close();

    const std::string cmd = std::string(VENV_PYTHON) + " visualize.py";
    std::system(cmd.c_str());
    return 0;
}
