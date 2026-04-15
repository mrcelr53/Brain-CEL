/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#include <braincel/SimulationWorker.h>
#include <braincel/uuid.h>

#include "spiking/Network.h"
#include "spiking/Neuron.h"
#include "spiking/Axon.h"
#include "spiking/Dendrite.h"
#include "spiking/Membrane.h"
#include "spiking/Synapse.h"
#include "spiking/Plasticity.h"
#include "spiking/Layer.h"
#include "functional/Function.h"
#include "core/Timed.h"
#include "core/Distribution.h"

#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>
#include <ranges>
#include <stdexcept>
#include <cassert>


using json = nlohmann::json;

static DistributionSampler sampler;

// Port helpers
static void createInputPorts(auto& node, const json& nodeParam, Network* net,
                              const json& inPortsParam,
                              std::unordered_map<std::string, std::vector<Timed*>>& inMap) {
    int numF32 = 0, numVecF32 = 0, numSoma = 0, numDendrite = 0;
    for (const auto& p : inPortsParam) {
        const std::string pt = p.value("port_type", std::string("Vector"));
        if (pt == "Scalar")   ++numF32;
        if (pt == "Vector")   ++numVecF32;
        if (pt == "Soma")     ++numSoma;
        if (pt == "Dendrite") ++numDendrite;
    }
    node.template reserveInputs<F32InputType>(numF32);
    node.template reserveInputs<VecF32InputType>(numVecF32);
    if constexpr (requires { node.reserveDendrites(0); })
        node.reserveDendrites(numDendrite);

    bool somaSet = false;
    for (const auto& p : inPortsParam) {
        const std::string inName  = p.value("name", std::string(""));
        const std::string portType= p.value("port_type", std::string("Vector"));
        const std::string portId  = p.value("id", std::string(""));

        if (portType == "Scalar") {
            auto& v = node.template emplaceInput<F32InputType>();
            v.setClassUuid(uuidToUint64Pair(portId));
            v.setName(inName);
            inMap[portId].push_back(&v);
        }
        else if (portType == "Vector") {
            auto& v = node.template emplaceInput<VecF32InputType>();
            v.setClassUuid(uuidToUint64Pair(portId));
            v.setName(inName);
            inMap[portId].push_back(&v);
        }
        else if (portType == "Soma") {
            if (somaSet) continue;
            if constexpr (requires { node.soma(); }) {
                auto& soma = node.soma();
                soma.setClassUuid(uuidToUint64Pair(portId));
                soma.setParent(&node);
                inMap[portId].push_back(&soma);
                somaSet = true;
            }
        }
        else if (portType == "Dendrite") {
            if constexpr (requires { node.setSynapticScaling(true);
                                     node.setScalingAdaption(0.0); }) {
                const auto& denParam = nodeParam.value("dendrite", json::object());
                const std::string denType = denParam.value("type", std::string(""));
                if (denType == "Passive") {
                    node.setSynapticScaling(denParam.value("synaptic_scaling", true));
                    node.setScalingAdaption(denParam.value("scaling_adaption", 0.005));
                } else if (denType == "Active") {
                    // TODO
                } else if (!denType.empty()) {
                    std::cerr << "[Build] Dendrite type '" << denType << "' not supported.\n";
                } else {
                    node.setSynapticScaling(false);
                    node.setScalingAdaption(0.0);
                }
            }
        }
    }
}

static void createOutputPorts(auto& node, const json& nodeParam, Network* net,
                               const json& outPortsParam,
                               std::unordered_map<std::string, std::vector<Timed*>>& outMap) {
    int numF32 = 0, numVecF32 = 0;
    for (const auto& p : outPortsParam) {
        const std::string pt = p.value("port_type", std::string("Vector"));
        if (pt == "Scalar") ++numF32;
        else                ++numVecF32;
    }
    node.template reserveOutputs<F32OutputType>(numF32);
    node.template reserveOutputs<VecF32OutputType>(numVecF32);

    bool axonSet = false;
    for (const auto& p : outPortsParam) {
        const std::string outName    = p.value("name",       std::string(""));
        const std::string portColor  = p.value("color",      std::string(""));
        const std::string portType   = p.value("port_type",  std::string("Vector"));
        const std::string portId     = p.value("id",         std::string(""));
        const bool isVisualized      = p.value("visualized", false);

        if (portType == "Scalar") {
            auto& v = node.template emplaceOutput<F32OutputType>();
            v.setClassUuid(uuidToUint64Pair(portId));
            v.setName(outName);
            outMap[portId].push_back(&v);
            if (isVisualized) {
                net->registerStateGetter<float>(
                   [&v]() { return v.value(); }, portId, outName, portColor);
            }
        }
        else if (portType == "Vector") {
            auto& v = node.template emplaceOutput<VecF32OutputType>();
            v.setClassUuid(uuidToUint64Pair(portId));
            v.setName(outName);
            outMap[portId].push_back(&v);
            if (isVisualized) {
                net->registerStateGetter<std::vector<float>>(
                   [&v]() { return v.value(); }, portId, outName, portColor);
            }
        }
        else if (portType == "Axon" || portType == "ExcitatoryAxon" || portType == "InhibitoryAxon") {
            if (axonSet) continue;
            if constexpr (requires { node.axon(); }) {
                auto& axon = node.axon();
                axon.setClassUuid(uuidToUint64Pair(portId));
                axon.setParent(&node);
                outMap[portId].push_back(&axon);

                const auto& axonParam  = nodeParam.value("axon", json::object());
                const std::string tx   = axonParam.value("transmitter", std::string("No Transmitter"));
                const double condSpeed = sampler.sample(axonParam["conduction_speed"], 0.2);
                const bool fixedLen    = axonParam.value("use_fixed_length", false);
                const double pathLen   = sampler.sample(axonParam["path_length"], 300.0);

                if      (tx == "Glutamate")      axon.setTransmitter(Glutamate);
                else if (tx == "GABA")           axon.setTransmitter(GABA);
                else if (tx == "No Transmitter") axon.setTransmitter(NoTransmitter);
                else std::cerr << "[Build] Unknown transmitter '" << tx << "'\n";

                if (fixedLen) {
                    const float delay = static_cast<float>(pathLen / condSpeed * 1e-3);
                    assert(!std::isinf(delay) && delay >= 0.f);
                    axon.setDelay(delay);
                }
                axonSet = true;
            }
        }
    }
}


static Timed* createConnection(AxonType* preAxon, SomaType* postSoma,
                                int index, const std::string& conId,
                                const json& conParam) {
    if (!preAxon || !postSoma) return nullptr;

    const auto& mainParam  = conParam.value("connection",    json::object());
    const auto& learnParam = conParam.value("learning",      json::object());
    const auto& preTerm    = conParam.value("pre_terminal",  json::object());
    const auto& postTerm   = conParam.value("post_terminal", json::object());

    const double weight       = sampler.sample(mainParam.value("weight",          json(1.0)),  1.0);
    const double delay        = sampler.sample(mainParam.value("delay",           json(0.0)),  0.0);
    const double learningRate = sampler.sample(learnParam.value("rate",           json(0.0)),  0.0);
    const double spikeDur     = sampler.sample(learnParam.value("spike_duration", json(0.01)), 0.01);
    const std::string rule    = learnParam.value("rule",             std::string("LPL"));
    const bool fwdTriggered   = learnParam.value("forward_triggered", true);
    const double minWeight    = sampler.sample(learnParam.value("min_weight",     json(0.0)),  0.0);
    const double maxWeight    = sampler.sample(learnParam.value("max_weight",     json(10.0)), 10.0);

    if (rule == "No Rule") {
        auto& con = preAxon->emplaceOutput<AxoSomaticStaticSynapseType>(*postSoma);
        con.setClassUuid(uuidToUint64Pair(conId));
        con.setWeight(weight); con.setDelay(delay); con.setLearningRate(0.);
        return &con;
    }

    auto& con = preAxon->emplaceOutput<AxoSomaticSynapseType>(*postSoma);
    con.setClassUuid(uuidToUint64Pair(conId));
    con.setWeight(weight); con.setMinWeight(minWeight); con.setMaxWeight(maxWeight);
    con.setLearningRate(learningRate);
    con.setDelay(delay);
    postSoma->setForwardTriggered(fwdTriggered);  // non-heterogeneous

    if (rule == "STDP" && fwdTriggered) {
        const double ap = sampler.sample(learnParam.value("stdp_a_plus",   json(1.0)),  1.0);
        const double am = sampler.sample(learnParam.value("stdp_a_minus",  json(1.0)),  1.0);
        const double mp = sampler.sample(learnParam.value("stdp_mu_plus",  json(0.0)),  0.0);
        const double mm = sampler.sample(learnParam.value("stdp_mu_minus", json(0.0)),  0.0);
        const double pj = sampler.sample(preTerm.value("stdp_jump_amp",    json(1.0)),  1.0);
        const double pt = sampler.sample(preTerm.value("stdp_jump_tau",    json(20.0)), 20.0);
        const double qj = sampler.sample(postTerm.value("stdp_jump_amp",   json(1.0)),  1.0);
        const double qt = sampler.sample(postTerm.value("stdp_jump_tau",   json(20.0)), 20.0);
        using Pre = PreRule<SparseSTDPf, AxonType>;
        using Post= PostRule<SparseSTDPf, SomaType>;
        using Con = ConnectionRule<SparseSTDPf, AxonType, SomaType, AxoSomaticSynapseType>;
        auto& pre  = preAxon->hasProcess<Pre>()  ? preAxon->process<Pre>()
                   : preAxon->emplaceProcess<Pre>(*preAxon, pj, pt);
        auto& post = postSoma->template hasProcess<Post>()? postSoma->template process<Post>()
                   : postSoma->template emplaceProcess<Post>(*postSoma, qj, qt);
        if (!con.template hasProcess<Con>())
            con.template emplaceProcess<Con>(con, *preAxon, *postSoma, pre, post, ap, am, mp, mm, maxWeight);
    }
    if (rule == "STDP") {
        const double ap = sampler.sample(learnParam.value("stdp_a_plus",  json(1.0)),  1.0);
        const double am = sampler.sample(learnParam.value("stdp_a_minus", json(1.0)),  1.0);
        const double mp = sampler.sample(learnParam.value("stdp_mu_plus", json(0.0)),  0.0);
        const double mm = sampler.sample(learnParam.value("stdp_mu_minus",json(0.0)),  0.0);
        const double pj = sampler.sample(preTerm.value("stdp_jump_amp",   json(1.0)),  1.0);
        const double pt = sampler.sample(preTerm.value("stdp_tau",        json(20.0)), 20.0);
        const double qj = sampler.sample(postTerm.value("stdp_jump_amp",  json(1.0)),  1.0);
        const double qt = sampler.sample(postTerm.value("stdp_tau",       json(20.0)), 20.0);
        using Pre = PreRule<SparseSTDP, AxonType>;
        using Post= PostRule<SparseSTDP, SomaType>;
        using Con = ConnectionRule<SparseSTDP, AxonType, SomaType, AxoSomaticSynapseType>;
        auto& pre  = preAxon->hasProcess<Pre>()  ? preAxon->process<Pre>()
                   : preAxon->emplaceProcess<Pre>(*preAxon, pj, pt);
        auto& post = postSoma->template hasProcess<Post>()? postSoma->template process<Post>()
                   : postSoma->template emplaceProcess<Post>(*postSoma, qj, qt);
        if (!con.template hasProcess<Con>())
            con.template emplaceProcess<Con>(con, *preAxon, *postSoma, pre, post, ap, am, mp, mm);
    }
    return &con;
}

static Timed* createConnection(AxonType* preAxon, DendriteType* postDendrite,
                                int index, const std::string& conId,
                                const json& conParam) {
    if (!preAxon || !postDendrite) return nullptr;
    auto& con = preAxon->emplaceOutput<AxoDendriticSynapseType>(*postDendrite);
    con.setClassUuid(uuidToUint64Pair(conId));
    return &con;
}

template <typename VariableType, typename OutputType, typename InputType>
static Timed* createConnection(OutputType* pre, InputType* post,
                                int index, const std::string& conId,
                                const json& conParam) {
    if (!pre || !post) return nullptr;
    if constexpr (OutputType::OutputBase::is_pointer_storage) {
        auto con = VariableType(pre->host(), *pre, *post);
        pre->template insertOutput<VariableType>(con);
        con.setClassUuid(uuidToUint64Pair(conId));
        con.setIndex(index);
        return &con;
    } else {
        auto& con = pre->template emplaceOutput<VariableType>(*post);
        con.setClassUuid(uuidToUint64Pair(conId));
        con.setIndex(index);
        return &con;
    }
}

// SimulationWorker

SimulationWorker::SimulationWorker() {
    net = new Network();
}
SimulationWorker::~SimulationWorker() {
    delete net;
}

void SimulationWorker::simulationChanged(const json& params) {
    if (params == simParams) return;
    simParams = params;
    flagSimulationChange_ = true;
}

void SimulationWorker::loadSimulationParams() {
    duration             = simParams.value("duration",          1000.0);
    timestep             = static_cast<float>(simParams.value("time_step",    1.0));
    sleepTime            = simParams.value("sleep_time",           0.0);
    warmupTime           = simParams.value("warmup_time",          0.0);
    globalLearningFactor = static_cast<float>(simParams.value("learning_factor", 1.0));
    randomDebugInput     = simParams.value("random_input",       false);
    batchSimulation      = simParams.value("batch_simulation",   false);
    device               = simParams.value("device", "GPU Compute");

    totalTicks = static_cast<int>(duration / timestep);

    for (const auto &view: net->neuronViews() | views::values)
        view->setVisualLimit(instancesPerModule);

    const auto& logP = simParams.value("logging", json::object());
    doLog     = logP.value("activated",    true);
    instancesPerModule   = logP.value("instances_per_node",     50);
    instanceAmount       = logP.value("instance_amount",        0.1);
    updateIntervalTicks  = std::max(1, static_cast<int>(logP.value("update_interval", 100.0) / timestep));
    updateIntervalSpikeIds = std::max(1, static_cast<int>(logP.value("update_interval_spike", 1.0) / timestep));
    dftBinSize           = logP.value("dft_bin_size",          1024);
    valueState           = logP.value("value_state", std::string("Spikes"));
    absActivity          = logP.value("abs_activity",           true);

    logSkips  = std::max(0, logP.value("skips", 100));
    logPrefix = logP.value("prefix", std::string("sim"));
}

std::map<int, std::map<int, float>> SimulationWorker::getConnections(
        const std::string& preGroup, const std::string& postGroup) {
    const auto out = net->getWeights(preGroup, postGroup);
    std::map<int, std::map<int, float>> data;
    for (const auto& [k, inner] : out)
        for (const auto& [k2, v] : inner)
            data[k][k2] = v;
    return data;
}

void SimulationWorker::build() {
    try {
        std::cout << "[SimulationWorker] Build started.\n";
        localBuild(buildParams);
        buildComplete_ = true;
        std::cout << "[SimulationWorker] Build finished.\n";
    } catch (const std::exception& e) {
        std::cerr << "[SimulationWorker] Build error: " << e.what() << "\n";
    }
}

void SimulationWorker::simulate() {
    try {
        if (!buildComplete_) build();
        std::cout << "[SimulationWorker] Simulation started.\n"
                  << "  Duration      : " << simParams.value("duration", 1000.0) << "\n"
                  << "  Time Step     : " << simParams.value("time_step", 1.0)   << "\n";
        localSimulate();
        std::cout << "[SimulationWorker] Simulation finished.\n";
    } catch (const std::exception& e) {
        std::cerr << "[SimulationWorker] Simulation error: " << e.what() << "\n";
    }
}

void SimulationWorker::pause() { localPause(); }
void SimulationWorker::reset() {
    localReset();
    tick_ = 0; biotime_ = 0.f; pause_ = false; buildComplete_ = false;
}
void SimulationWorker::clear() { localClear(); buildComplete_ = false; }


bool SimulationWorker::localBuild(const json& remoteBuildParams) {
    const auto startTime = std::chrono::high_resolution_clock::now();

    std::random_device rd;
    std::mt19937 gen(rd());

    net->clear();

    const auto& sceneParams = remoteBuildParams.value("scene", json::object());
    const auto& nodeParams  = sceneParams.value("nodes",       json::array());
    const auto& conParams   = sceneParams.value("connections", json::array());

    // Count instances
    int numNodeInstances = 0, numNeuronInstances = 0;
    for (const auto& nodeArg : nodeParams) {
        const auto& mp   = nodeArg.value("node", json::object());
        const int number = mp.value("number", 1);
        if (mp.value("type", std::string("")) == "Neuron") {
            numNeuronInstances += number;
        }
        else                                                numNodeInstances   += number;
    }

    std::cout << "[Build] Reserving — Nodes: " << numNodeInstances
              << "  Neurons: " << numNeuronInstances << "\n";

    auto& nodes   = net->nodes();
    auto& neurons = net->neurons();
    nodes.reserve<FunctionType>(numNodeInstances);
    neurons.reserve<NeuronType>(numNeuronInstances);

    std::unordered_map<std::string, std::vector<Timed*>> nodeMap, inMap, outMap, conMap;

    int instanceIdCounter = 0;
    int vizViewIdCounter  = 1;

    std::cout << "[Build] (1/2) Initializing nodes & neurons...\n";

    for (const auto& nodeArg : nodeParams) {
        const auto& inPortsParam  = nodeArg.value("input",  json::array());
        const auto& outPortsParam = nodeArg.value("output", json::array());
        const std::string nodeId  = nodeArg.value("id", std::string(""));
        const auto& mainParam     = nodeArg.value("node", json::object());
        const std::string name    = mainParam.value("name",   std::string(""));
        const std::string type    = mainParam.value("type",   std::string(""));
        const std::string color   = mainParam.value("color",  std::string(""));
        const int number          = mainParam.value("number", 1);
        const double firingRateTau= mainParam.value("firing_rate_tau", 10000.0);

        if (type == "Neuron") {
            auto* view = net->addNeuronView(name);
            view->setColor(color);
            view->setClassUuid(uuidToUint64Pair(nodeId));
            view->setId(vizViewIdCounter);
            view->setName(name);
            view->setClassName("Neuron View");
            view->setVisualized(true);

            const auto& memParam       = nodeArg.value("membrane", json::object());
            const std::string memType  = memParam.value("type", std::string("No Membrane"));
            const double baselineFire  = memParam.value("baseline_fire", 0.0);
            const double refractPeriod = memParam.value("refractory_period", 0.0);

            for (int i = 0; i < number; ++i) {
                auto& neuron = neurons.emplace<NeuronType>();
                view->insert<NeuronType>(&neuron);
                nodeMap[nodeId].push_back(&neuron);

                neuron.setId(instanceIdCounter++);
                neuron.setClassName(name);
                neuron.setClassUuid(uuidToUint64Pair(nodeId));
                neuron.setSimulated(true);
                neuron.setFiringRateTau(firingRateTau);

                if (memType == "LIF" || memType == "Leaky Integrate & Fire") {
                    auto resistance        = sampler.sample(memParam["resistance"],        10.0);
                    auto tauPotential      = sampler.sample(memParam["tau_potential"],     20.0);
                    auto restingPotential  = sampler.sample(memParam["resting_potential"],-70.0);
                    auto reversalPotential = sampler.sample(memParam["reversal_potential"],-60.0);
                    auto threshold         = sampler.sample(memParam["threshold"],        -50.0);
                    auto thresholdAdapt    = sampler.sample(memParam["threshold_adapt"],    1.0);
                    auto tauAdapt          = sampler.sample(memParam["tau_adapt"],          2.0);
                    auto refractoryPeriod  = sampler.sample(memParam["refractory_period"], 0.);
                    auto &membrane = neuron.emplaceMembrane<LifMembrane>(threshold, tauPotential, restingPotential,
                                                                         resistance,
                                                                         reversalPotential, baselineFire, thresholdAdapt,
                                                                         tauAdapt, refractoryPeriod);
                    membrane.setSeed(neuron.id());
                }
                else if (memType == "Izhikevich") {
                    const double threshold  = memParam.value("threshold", 30.0);
                    const double a          = memParam.value("a",          0.02);
                    const double b          = memParam.value("b",          0.2);
                    const double c          = memParam.value("c",         -65.0);
                    const double d          = memParam.value("d",          8.0);
                    auto& membrane = neuron.emplaceMembrane<IzhMembrane>(
                        threshold, a, b, c, d, -70., 13., baselineFire);
                    membrane.setSeed(neuron.id());
                }
                else if (memType == "No Membrane") {
                    neuron.setClampedPoissonRate(baselineFire);
                    neuron.setSimulated(false);
                }
                else {
                    std::cerr << "[Build] Membrane type '" << memType << "' not supported.\n";
                    neuron.setSimulated(false);
                }

                createInputPorts(neuron, nodeArg, net, inPortsParam, inMap);
                createOutputPorts(neuron, nodeArg, net, outPortsParam, outMap);
            }
        }
        else if (type == "Function") {
            auto* view = net->addNodeView(name);
            view->setColor(color);
            view->setId(vizViewIdCounter);
            view->setClassUuid(uuidToUint64Pair(nodeId));
            view->setName(name);
            view->setClassName("Node View");

            const auto& scriptParam  = nodeArg.value("script", json::object());
            const std::string path   = scriptParam.value("path", std::string(""));
            const std::string script = scriptParam.value("text", std::string(""));

            auto& fn = nodes.emplace<FunctionType>();
            view->insert<FunctionType>(&fn);
            fn.setId(instanceIdCounter++);
            fn.setClassName(name);
            fn.setClassUuid(uuidToUint64Pair(nodeId));
            fn.setSimulated(true);
            fn.setScript(path.empty() ? script : loadStringFromFile(path));
            nodeMap[nodeId].push_back(&fn);

            createInputPorts(fn,  nodeArg, net, inPortsParam,  inMap);
            createOutputPorts(fn, nodeArg, net, outPortsParam, outMap);
        }
        else {
            std::cerr << "[Build] Node type '" << type << "' not supported.\n";
        }
        vizViewIdCounter++;
    }

    neurons.reserveUpdateStorage<NeuronType>(0.2f);
    neurons.reserveActive();
    neurons.clearActive();

    // Pre-count connections
    int totalConnectionInstances = 0;
    std::unordered_map<std::string, int> outConPerPortCounts;
    std::unordered_map<std::string, int> numOutConPerPortByConnection;
    std::unordered_map<std::string, int> inPortConCounts;

    for (const auto& conArg : conParams) {
        const auto& cp       = conArg.value("params",     json::object());
        const auto& mp       = cp.value("connection",     json::object());
        const std::string conId  = conArg.value("id",   std::string(""));
        const std::string fromId = conArg.value("from", std::string(""));
        const std::string toId   = conArg.value("to",   std::string(""));

        const auto& fromPorts = outMap.count(fromId) ? outMap.at(fromId) : std::vector<Timed*>{};
        const auto& toPorts   = inMap.count(toId)    ? inMap.at(toId)    : std::vector<Timed*>{};

        const double density  = fromPorts.size() > 1 ? mp.value("density", 1.0) : 1.0;
        const int numOut      = toPorts.size() > 1
                                ? static_cast<int>(density * toPorts.size()) : 1;
        outConPerPortCounts[fromId]           += numOut;
        numOutConPerPortByConnection[conId]    = numOut;

        for (auto* p : fromPorts) totalConnectionInstances += numOut;
        for (auto* p : toPorts)
            inPortConCounts[uint64PairToUuid(p->classUuid())] +=
                static_cast<int>(fromPorts.size());
    }

    std::cout << "[Build] (2/2) Setting up connections & synapses...\n";
    int conCount = 0;

    for (const auto& conArg : conParams) {
        const auto& conParam = conArg.value("params",     json::object());
        const auto& mainParam= conParam.value("connection", json::object());
        const bool shuffled  = mainParam.value("shuffled", true);
        const std::string mode = mainParam.value("mode", std::string("Global"));
        const double localRadius = mainParam.value("radius", 0.2);

        const bool   useSeed = mainParam.value("use_seed", false);
        const uint   seed    = useSeed
                               ? static_cast<uint>(mainParam.value("seed", 42))
                               : rd();
        sampler.setSeed(seed);
        gen.seed(seed);

        const std::string conId  = conArg.value("id",   std::string(""));
        const std::string fromId = conArg.value("from", std::string(""));
        const std::string toId   = conArg.value("to",   std::string(""));

        auto fromPorts = outMap.count(fromId) ? outMap.at(fromId) : std::vector<Timed*>{};
        auto toPorts   = inMap.count(toId)    ? inMap.at(toId)    : std::vector<Timed*>{};
        const int numOutPerNode = std::min(numOutConPerPortByConnection[conId],
                                           static_cast<int>(toPorts.size()));

        int fromPortIdx = 0;
        for (auto* fromPort : fromPorts) {
            auto* fromF32    = dynamic_cast<F32OutputType*>(fromPort);
            auto* fromVecF32 = dynamic_cast<VecF32OutputType*>(fromPort);
            auto* fromAxon   = dynamic_cast<AxonType*>(fromPort);

            if (fromF32)    { fromF32->reserveOutputs<F32VariableType>(numOutPerNode * 10);
                              fromF32->reserveOutputs<F32VecF32VariableType>(numOutPerNode * 10); }
            if (fromVecF32) { fromVecF32->reserveOutputs<VecF32VariableType>(numOutPerNode * 10);
                              fromVecF32->reserveOutputs<VecF32F32VariableType>(numOutPerNode * 10); }
            if (fromAxon)   { fromAxon->reserveOutputs<AxoSomaticStaticSynapseType>(numOutPerNode * 10);
                              fromAxon->reserveOutputs<AxoSomaticSynapseType>(numOutPerNode * 10); }

            std::vector<Timed*> eligible;
            if (mode == "Local") {
                const double normPos = static_cast<double>(fromPortIdx) / fromPorts.size();
                const int radius     = static_cast<int>(localRadius * toPorts.size());
                const int center     = static_cast<int>(normPos * toPorts.size());
                for (int i = -radius; i <= radius; ++i) {
                    int idx = center + i;
                    if (idx >= 0 && idx < static_cast<int>(toPorts.size()))
                        eligible.push_back(toPorts[idx]);
                }
            } else {
                eligible = toPorts;
            }
            if (shuffled) std::ranges::shuffle(eligible, gen);

            const int targets = std::min(numOutPerNode, static_cast<int>(eligible.size()));
            for (int toIdx = 0; toIdx < targets; ++toIdx) {
                auto* toPort     = eligible[toIdx];
                auto* toF32      = dynamic_cast<F32InputType*>(toPort);
                auto* toVecF32   = dynamic_cast<VecF32InputType*>(toPort);
                auto* toSomaFT     = dynamic_cast<SomaType*>(toPort);
                auto* toSomaBiT     = dynamic_cast<SomaType*>(toPort);
                auto* toDendrite = dynamic_cast<DendriteType*>(toPort);

                const int numIn = inPortConCounts[uint64PairToUuid(toPort->classUuid())];
                if (toVecF32) toVecF32->resize(numIn);

                Timed* connection = nullptr;
                Timed* tmp = nullptr;
                if      ((tmp = createConnection(fromAxon, toSomaFT,     toIdx, conId, conParam))) connection = tmp;
                else if ((tmp = createConnection(fromAxon, toSomaBiT,     toIdx, conId, conParam))) connection = tmp;
                else if ((tmp = createConnection(fromAxon, toDendrite, toIdx, conId, conParam))) connection = tmp;
                else if ((tmp = createConnection<F32VariableType>       (fromF32,    toF32,    toIdx, conId, conParam))) connection = tmp;
                else if ((tmp = createConnection<VecF32F32VariableType> (fromVecF32, toF32,    toIdx, conId, conParam))) connection = tmp;
                else if ((tmp = createConnection<F32VecF32VariableType> (fromF32,    toVecF32, fromPortIdx, conId, conParam))) connection = tmp;
                else if ((tmp = createConnection<VecF32VariableType>    (fromVecF32, toVecF32, toIdx, conId, conParam))) connection = tmp;
                else std::cerr << "[Build] Unknown connection type.\n";

                if (connection) conMap[conId].push_back(connection);
            }
            conCount += numOutPerNode;
            fromPortIdx++;
        }
    }

    const auto now = std::chrono::high_resolution_clock::now();
    lastBuildTime = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count());

    std::cout << "[Build] Finished in " << lastBuildTime / 1000.0 << " s\n"
              << net->stats() << "\n";
    return true;
}


void SimulationWorker::localSimulate() {
    loadSimulationParams();

    net->setTimestep(timestep);

    auto& neurons = net->neurons();

    if (doLog) {
        net->generateSubsampledConnectome(instanceAmount);
        neurons.setCallbackMembrane(true);
    }

    if (device == "GPU Compute") {
        neurons.setCuda(true);
    }

    bool isWarmup = warmupTime > 0;
    net->setGlobalLearningFactor(isWarmup ? 0.f : globalLearningFactor);

    if (doLog) {
        if (batchSimulation) logger.initializeBatchDirectory(logPrefix);
        logger.initializeDirectory(logPrefix);
        logger.logParams("build", buildParams);
        logger.logParams("simulation", simParams);
    }

    const auto startRealtime = std::chrono::high_resolution_clock::now();
    auto endIntervalRealtime = startRealtime;
    double walltime = 0.0, rtf = 0.0;

    std::vector<std::vector<float>> batchData;
    std::vector<std::tuple<float, int, int>> batchScatterData;
    std::vector<float> batchActivity;
    std::vector<std::vector<int>> batchSpikeIds;

    std::cout << "[Simulate] Running...\n";

    for (; tick_ < totalTicks; ++tick_) {
        net->cycle();
        net->tick();

        if (doLog) {
            if (valueState == "Spike Times") {
                for (const auto& el : net->spikeTimes())
                    batchScatterData.push_back(el);
            }
            else {
                std::vector<float> values;
                if      (valueState == "Potentials")  { auto v = net->potentials();  values.assign(v.begin(), v.end()); }
                else if (valueState == "Weights")     { auto v = net->weights();     values.assign(v.begin(), v.end()); }
                else                                  { auto v = net->spikes();      values.assign(v.begin(), v.end()); }
                batchData.push_back(std::move(values));
            }
            const auto numSpk = net->numSpikes();
            const auto act = net->activity();
            batchActivity.push_back(absActivity ? static_cast<float>(numSpk) : act);

            const bool updateDue = (tick_ % updateIntervalTicks == updateIntervalTicks - 1);
            const bool lastTick  = (tick_ + 1 >= totalTicks);
            if (updateDue || pause_ || lastTick) {
                endIntervalRealtime = std::chrono::high_resolution_clock::now();
                walltime = static_cast<double>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        endIntervalRealtime - startRealtime).count());
                rtf = walltime / biotime_;

                auto [connectome, distribution,
                      numBins, binMin, binMax, binStep] = net->gatherSubsampledConnectome();

                if (logSkips > 0 && loggingSkip == logSkips) {
                    if (!batchData.empty())
                        logger.logDataBatch(tick_, biotime_, batchData, valueState);
                    if (!batchScatterData.empty())
                        logger.logScatterBatch(tick_, biotime_, batchScatterData, valueState);
                    logger.logActivityBatch(tick_, biotime_, batchActivity);
                    logger.logSpikeIdBatch(tick_, biotime_, batchSpikeIds);
                    logger.logConnectome(tick_, biotime_,
                        std::vector(connectome.begin(), connectome.end()));
                    logger.logAbsDistribution(tick_, biotime_, distribution,
                                              binMin, binMax, binStep, "Weights");
                    logger.logFloatStates(tick_, biotime_, biotime_,
                        net->gatherStateGetters<float>(), "Scalars");

                    nlohmann::json simInfo;
                    simInfo["tick"]        = tick_;
                    simInfo["biotime"]     = biotime_;
                    simInfo["walltime"]    = walltime;
                    simInfo["total_ticks"] = totalTicks;
                    simInfo["rtf"]         = rtf;
                    logger.logSimInfo(tick_, biotime_, simInfo);
                    loggingSkip = 0;
                } else {
                    ++loggingSkip;
                }

                batchData.clear();
                batchActivity.clear();
                batchScatterData.clear();
            }
        }

        biotime_ += timestep;

        if (pause_) break;

        if (flagSimulationChange_) {
            loadSimulationParams();
            flagSimulationChange_ = false;
        }
        if (sleepTime > 0.0)
            std::this_thread::sleep_for(
                std::chrono::duration<double, std::milli>(sleepTime));
        if (isWarmup && warmupTime < biotime_) {
            isWarmup = false;
            net->setGlobalLearningFactor(globalLearningFactor);
        }
    }

    neurons.setCuda(false);

    const auto endTime = std::chrono::high_resolution_clock::now();
    walltime = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startRealtime).count());
    rtf = walltime / duration;

    if (logger.isInitialized()) {
        nlohmann::json simInfo;
        simInfo["tick"]      = tick_ + 1;
        simInfo["buildtime"] = lastBuildTime;
        simInfo["biotime"]   = biotime_;
        simInfo["walltime"]  = walltime;
        simInfo["rtf"]       = rtf;
        logger.logSimInfo(tick_, biotime_, simInfo);
    }

    std::cout << "--------- SIMULATION STATS -------\n"
              << "  Nodes:      " << net->numNodes()      << "\n"
              << "  Parameters: " << net->numVariables() << "\n"
              << "  Neurons:    " << net->numNeurons()    << "\n"
              << "  Synapses:   " << net->numSynapses()   << "\n"
              << "  Built:      " << lastBuildTime        << " ms\n"
              << "  Simulated:  " << duration             << " ms\n"
              << "  Elapsed:    " << walltime             << " ms\n"
              << "  RTF:        " << rtf                  << "\n"
              << "----------------------------------\n";
}

void SimulationWorker::localPause() { pause_ = true; }
void SimulationWorker::localReset() { net->reset(); }
void SimulationWorker::localClear() { delete net; net = new Network(); }