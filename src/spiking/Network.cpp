/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#include "Network.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;


Network::Network(const std::string &name)
    : Host(name),
      nodes_(this, "Node Population"),
      neurons_(this, "Neuron Population") {
}

// Simulation
void Network::cycle() {
    neurons_.clearSpikes();
    for (auto* view : neuronViews_ | views::values) { view->clearSpikes(); }
}
void Network::tick() {
    // --- Clock start ---
    const auto start = std::chrono::high_resolution_clock::now();

    // Activate all neurons
    updateQueue()->tick();
    nodes_.update();
    neurons_.update();

    // Propagate the information
    pushQueue()->tick();
    nodes_.push();
    neurons_.push();

    // --- Clock end ---
    const std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;

    // Set stats and finish off this tick
    advance(duration.count());
}
void Network::clear() {
    reset();
    if constexpr (nodes_.PointerStorage) {
    nodes_.forEach([](const auto& node) {
           delete node;
        });
    }
    nodes_.clear();

    if constexpr (neurons_.PointerStorage) {
        neurons_.forEach([](const auto& neuron) {
           delete neuron;
        });
    }
    neurons_.clear();
    neuronViews_.clear();
    neuronIdOffset_ = 0;
}
void Network::reset() {
    nodes_.reset();
    nodes_.forEach([](auto& node) {
        node.reset();
    });
    neurons_.reset();
    neurons_.forEach([](auto& neuron) {
        neuron.reset();
    });
    resetTick();
}

json Network::neuronViewSerialize() const  {
    auto data = json::array();
    for (auto* view: neuronViews_ | views::values) {
        data.push_back(view->serialize());
    }
    return data;
}
json Network::neuronViewColors() const  {
    auto colors = json();
    for (auto& [key, view]: neuronViews_) {
        colors[key] = view->color();
    }
    return colors;
}

void Network::print(const bool printSpikes, const int spacing) {
    const int numDigits = (currentTick() == 0) ? 1 : static_cast<int>(log10(currentTick())) + 1;
    const int spaceWidth = spacing - numDigits - 2;

    // Print the tick and speedup factor
    std::cout << name() << ": ";
    printf("T(%.2fx)+%lu %*s", speedup(), currentTick(), spaceWidth, "");

    int i = 0;
    int j = 0;
    std::cout << "Groups: [ ";
    for (const auto& [fst, view] : neuronViews_) {
        std::string groupName = fst;
        if (groupName.compare(0, 2, "__") != 0) {
            const double activity = view->activity();
            const int spikes = view->numSpikes();

            i = 0;
            // Format the group information with consistent spacing and rounding
            printf("%-3s(%-1i)", groupName.substr(0, 3).c_str(), spikes);
            if (printSpikes) {
                std::cout << ": ";
                view->forEach([&i](auto& module) -> bool {
                    if (module.spike()) { std::cout << "#"; }  //"●";
                    else { std::cout << "."; }  //"◯";
                    i++;
                    if (i > 25) { return false; }
                    return true;
                });
            }

            std::cout << " ";
            j++;
        }
    }
    std::cout << "]\n";
}
std::string Network::stats() {
    std::stringstream sts;
    sts << "Network \"" << name() << "\":\n";
    sts << "- Nodes: " << numNodes() << "\n";
    sts << "- Variables: " << numVariables() << "\n";
    sts << "- Neurons: " << numNeurons() << "\n";
    if (numNeurons() > 0) {
        for (const auto& [fst, view] : neuronViews_) {
            sts << "  - " << fst << ": " << view->size() << "\n";
        }
    }
    sts << "- Synapses: " << numSynapses() << "\n";
    return sts.str();
}


void Network::generateSubsampledConnectome(const double amount) {
    neurons_.cacheSubsampleConnectome(amount);
}
std::unordered_map<int, std::unordered_map<int, float>> Network::getWeights(const std::string& preView,
                                                                            const std::string& postView) const {
    if (neuronViews_.contains(preView) && neuronViews_.contains(postView)) {
        const auto* g1 = neuronViews_.at(preView);
        const auto* g2 = neuronViews_.at(postView);
        return getWeights(g1, g2);
    }
    return {};
}
std::unordered_map<int, std::unordered_map<int, float>> Network::getWeights(const NeuronView<NeuronPack>* preView, const NeuronView<NeuronPack>* postView) const {
    std::unordered_map<int, std::unordered_map<int, float>> con;
    preView->forEach([&postView, &con](auto& pre) {
        std::unordered_map<int, float> preCon;
        const auto preId = pre.id();
        pre.axon().forEachOutput([&postView, &preCon, &preId](auto& syn) {
            const auto postId = syn.post().id();
            if (postView->contains(postId)) {
                const auto weight = syn.weight();
                preCon[postId] = weight;
            }
        });
        con[preId] = preCon;
    });
    return con;
}


size_t Network::numVariables() const  {
    size_t count = 0;
    nodes_.forEach([&count](auto& node) {
        count += node.numOutputConnections();
    });
    return count;
}
size_t Network::numSynapses() const {
    size_t count = 0;
    neurons_.forEach([&count](auto& neuron) {
        count += neuron.axon().numOutputs();
    });
    return count;
}

std::vector<int> Network::spikeIds() const {
    return neurons_.spikes();
}
std::vector<std::tuple<float, int, int>> Network::spikeTimes() const {
    std::vector<std::tuple<float, int, int>> tms;
    for (const auto* view : neuronViews_ | views::values) {
        if (!view->isVisualized()) { continue; }
        auto visualSpikes = view->visualSpikeTimes();
        tms.insert(tms.end(), visualSpikes.begin(), visualSpikes.end());
    }
    return tms;
}
std::vector<float> Network::spikes() const {
    std::vector<float> spikes;
    //spikes.reserve(groups.size() * neuronsPerGroup);
    for (const auto* view : neuronViews_ | views::values) {
        if (!view->isVisualized()) { continue; }
        auto visualSpikes = view->visualSpikes();
        spikes.insert(spikes.end(), visualSpikes.begin(), visualSpikes.end());
    }
    return spikes;
}
std::vector<float> Network::potentials() const {
    std::vector<float> potentials;
    //spikes.reserve(groups.size() * neuronsPerGroup);
    for (const auto* view : neuronViews_ | views::values) {
        auto visualPotentials = view->visualPotentials();
        potentials.insert(potentials.end(), visualPotentials.begin(), visualPotentials.end());
    }
    return potentials;
}
std::vector<float> Network::weights(const bool weightChange, const int step) const {
    std::vector<float> weights;
    return weights;
}