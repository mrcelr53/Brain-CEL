/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;


#include "../../include/braincel/build.h"

struct StateGetter {
    std::string name;
    std::string color = "";
    std::function<std::any()> getter;
};

class Network final : public Host {
public:
    explicit Network(const std::string& name = "Network");
    ~Network() override = default;

    void cycle();
    void tick();
    void clear();
    void reset();

    void setNodeIdOffset(const int offset) { nodeIdOffset_ = offset; }
    NodePopulation<NodePack>& nodes() { return nodes_; }
    void reserveNodeViews(const size_t numViews) { nodeViews_.reserve(numViews); }
    NodeView<NodePack>* nodeView(const std::string& key) const { return nodeViews_.at(key); }
    template <typename... Args>
    NodeView<NodePack>* addNodeView(const std::string& key, Args&&... args) {
        // Remove old entry if it exists
        const auto old = nodeViews_.find(key);
        if (old != nodeViews_.end()) { nodeViews_.erase(old); }

        auto* view = new NodeView<NodePack>(this, std::forward<Args>(args)...);
        nodeViews_[key] = view;
        return view;
    }

    void setNeuronIdOffset(const int offset) { neuronIdOffset_ = offset; }
    NeuronPopulation<NeuronPack>& neurons() { return neurons_; }
    void reserveNeuronViews(const size_t numViews) { neuronViews_.reserve(numViews); }
    NeuronView<NeuronPack>* neuronView(const std::string& key) const { return neuronViews_.at(key); }
    template <typename... Args>
    NeuronView<NeuronPack>* addNeuronView(const std::string& key, Args&&... args) {
        // Remove old entry if it exists
        const auto old = neuronViews_.find(key);
        if (old != neuronViews_.end()) { neuronViews_.erase(old); }

        auto* view = new NeuronView<NeuronPack>(this, std::forward<Args>(args)...);
        neuronViews_[key] = view;
        return view;
    }
    auto& neuronViews() { return neuronViews_; }

    json connectionMetaSerialize() const;
    json neuronViewSerialize() const;
    json neuronViewColors() const;
    json groupData() const;

    void print(bool printSpikes = false, int spacing = 10);
    std::string stats();

    json simulationStats() const {
        auto data = json();
        return data;
    }
    int numSpikes() const { return neurons_.numSpikes(); }
    float activity() const { return neurons_.activity(); }

    void generateSubsampledConnectome(double amount = 0.1);
    auto gatherSubsampledConnectome() const{
        return neurons_.getSubsampleConnectome();
    }

    std::unordered_map<int, std::unordered_map<int, float>> getWeights(const std::string& preView,
                                                                       const std::string& postView) const;
    std::unordered_map<int, std::unordered_map<int, float>> getWeights(const NeuronView<NeuronPack>* preView,
                                                                       const NeuronView<NeuronPack>* postView) const;

    float speedup() const { return speedup_; }

    size_t numNodes() const { return nodes_.size(); }
    size_t numVariables() const;
    size_t numNeurons() const { return neurons_.size(); }
    size_t numSynapses() const;

    std::vector<int> spikeIds() const;
    std::vector<std::tuple<float, int, int>> spikeTimes() const;
    std::vector<float> spikes() const;
    std::vector<float> potentials() const;
    std::vector<float> weights(bool weightChange = false, int step = 100) const;

    // Variable getters
    template <typename SignalType>
    void registerStateGetter(std::function<SignalType()> valueGetter,
                             const std::string& portId, const std::string& name,
                             const std::string& color = "") {
        stateGetters_[portId] = StateGetter{name, color,
            [getter = std::move(valueGetter)]() -> std::any { return getter(); }
        };
    }
    void unregisterStateGetter(const std::string& portId) {
        stateGetters_.erase(portId);
    }
    template <typename SignalType>
    std::unordered_map<std::string, SignalType> gatherStateGetters() const {
        std::unordered_map<std::string, SignalType> result;
        for (const auto &entry: stateGetters_ | views::values) {
            const auto& [name, color, getter] = entry;
            try { result.emplace(name, std::any_cast<SignalType>(getter())); }
            catch (const std::bad_any_cast&) {}  // Skip
        }
        return result;
    }

    std::unordered_map<std::string, std::any> gatherAllStateGetters() const {
        std::unordered_map<std::string, std::any> result;
        for (const auto &entry: stateGetters_ | views::values) {
            const auto& [name, color, getter] = entry;
            result[name] = getter();
        }
        return result;
    }
    json gatherStateGettersMeta() const {
        json result = json::object();
        for (const auto& entry : stateGetters_ | views::values) {
            result[entry.name] = {
                {"color", entry.color}
            };
        }
        return result;
    }

private:
    int nodeIdOffset_ = 0;
    NodePopulation<NodePack> nodes_;
    std::unordered_map<std::string, NodeView<NodePack>*> nodeViews_{};

    int neuronIdOffset_ = 0;
    NeuronPopulation<NeuronPack> neurons_;
    std::unordered_map<std::string, NeuronView<NeuronPack>*> neuronViews_{};

    bool distributionsEnabled_ = false;
    float speedup_ = 0.f;

    // Variable getter for collecting port values
    std::unordered_map<std::string, StateGetter> stateGetters_;
};

#endif // NETWORK_H