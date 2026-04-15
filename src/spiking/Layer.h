/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef NEURON_COLLECTION_H
#define NEURON_COLLECTION_H

#include <random>
#include <list>
#include <curand_kernel.h>
#include <regex>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

#include "core/Timed.h"
#include "core/View.h"
#include "core/Partition.h"

#include "spiking/cuda/NeuronGroup.cuh"

#include "Neuron.h"
#include "Synapse.h"
#include "Plasticity.h"


template <typename MemberTypePack>
class NodeView final : public View<MemberTypePack> {
    friend class View<MemberTypePack>;

public:
    using ViewBase = View<MemberTypePack>;
    explicit NodeView(Host* host, const std::string& name = "Node View")
        : Timed(host), ViewBase(host), name_(name) {}
    ~NodeView() override = default;

    NodeView(NodeView&&) noexcept = default;
    NodeView& operator=(NodeView&&) noexcept = default;

    explicit NodeView(const NodeView&) = delete;
    NodeView& operator=(const NodeView&) = delete;

    json serialize() const {
        auto data = json();
        data["id"] = this->id();
        data["name"] = name();
        data["color"] = json::array({ color_[0], color_[1], color_[2] });
        data["number"] = this->size();
        return data;
    }

    // Visualization methods
    std::string name() { return name_; }
    void setName(const std::string& name) { name_ = name; }

    // Visualization methods
    std::string color() const {
        // Format the string as rgb(r, g, b)
        return "rgb(" + std::to_string(color_[0]) + ", " + std::to_string(color_[1]) + ", " + std::to_string(color_[2]) + ")";
    }
    void setColor(const int r, const int g, const int b) { color_[0] = r, color_[1] = g, color_[2] = b; }
    void setColor(const std::string& color) {
        const std::regex hexRegex("^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$");
        const std::regex rgbRegex(R"(^rgb\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)$)");
        std::smatch match;

        if (std::regex_match(color, match, hexRegex)) {
            // Handle hex color
            std::string hex = match[1].str();

            if (hex.size() == 3) {
                // Convert 3-digit hex to 6-digit hex
                hex = std::string(1, hex[0]) + std::string(1, hex[0]) +
                      std::string(1, hex[1]) + std::string(1, hex[1]) +
                      std::string(1, hex[2]) + std::string(1, hex[2]);
            }

            // Convert hex to RGB values
            color_[0] = std::stoi(hex.substr(0, 2), nullptr, 16);
            color_[1] = std::stoi(hex.substr(2, 2), nullptr, 16);
            color_[2] = std::stoi(hex.substr(4, 2), nullptr, 16);
        } else if (std::regex_match(color, match, rgbRegex)) {
            // Handle rgb(r, g, b) color
            color_[0] = std::stoi(match[1].str());
            color_[1] = std::stoi(match[2].str());
            color_[2] = std::stoi(match[3].str());
        } else {
            throw std::invalid_argument("Invalid color format");
        }
    }

private:
    std::string name_;
    uint8_t color_[3] = {50, 50, 50};
};

template <typename MemberTypePack>
class NodePopulation final : public Partition<NodePopulation<MemberTypePack>, MemberTypePack> {
    friend class Partition<NodePopulation, MemberTypePack>;

public:
    using PartitionBase = Partition<NodePopulation, MemberTypePack>;

    explicit NodePopulation(Host* host, const std::string& name = "Node Population")
        : Timed(host), PartitionBase(host), name_(name) {}
    ~NodePopulation() override = default;

    NodePopulation(NodePopulation&&) noexcept = default;
    NodePopulation& operator=(NodePopulation&&) noexcept = default;

    explicit NodePopulation(const NodePopulation&) = delete;
    NodePopulation& operator=(const NodePopulation&) = delete;

    template <typename NodeType>
    NodeType& emplace() {
        return static_cast<PartitionBase*>(this)->template emplace<NodeType>(this->host());
    }
    template <typename NodeType, typename... Args>
    NodeType& emplace(Args&&... args) {
        return static_cast<PartitionBase*>(this)->template emplace<NodeType>(this->host(), std::forward<Args>(args)...);
    }

    // Visualization methods
    std::string name() { return name_; }
    void setName(const std::string& name) { name_ = name; }

    // Visualization methods
    std::string color() const {
        // Format the string as rgb(r, g, b)
        return "rgb(" + std::to_string(color_[0]) + ", " + std::to_string(color_[1]) + ", " + std::to_string(color_[2]) + ")";
    }
    void setColor(const int r, const int g, const int b) { color_[0] = r, color_[1] = g, color_[2] = b; }
    void setColor(const std::string& color) {
        const std::regex hexRegex("^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$");
        const std::regex rgbRegex(R"(^rgb\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)$)");
        std::smatch match;

        if (std::regex_match(color, match, hexRegex)) {
            // Handle hex color
            std::string hex = match[1].str();

            if (hex.size() == 3) {
                // Convert 3-digit hex to 6-digit hex
                hex = std::string(1, hex[0]) + std::string(1, hex[0]) +
                      std::string(1, hex[1]) + std::string(1, hex[1]) +
                      std::string(1, hex[2]) + std::string(1, hex[2]);
            }

            // Convert hex to RGB values
            color_[0] = std::stoi(hex.substr(0, 2), nullptr, 16);
            color_[1] = std::stoi(hex.substr(2, 2), nullptr, 16);
            color_[2] = std::stoi(hex.substr(4, 2), nullptr, 16);
        } else if (std::regex_match(color, match, rgbRegex)) {
            // Handle rgb(r, g, b) color
            color_[0] = std::stoi(match[1].str());
            color_[1] = std::stoi(match[2].str());
            color_[2] = std::stoi(match[3].str());
        } else {
            throw std::invalid_argument("Invalid color format");
        }
    }

private:
    std::string name_;
    uint8_t color_[3] = {50, 50, 50};
};


/**
 * An Assembly is a selection of neurons, where each Neuron can belong to multiple overlapping Assemblies
 */
template <typename MemberTypePack>
class NeuronView final : public View<MemberTypePack> {
    friend class View<MemberTypePack>;

public:
    using ViewBase = View<MemberTypePack>;

    explicit NeuronView(Host* host, const std::string& name = "Neuron View")
        : Timed(host), ViewBase(host), rng_(std::random_device{}()), name_(name) {}
    ~NeuronView() override = default;

    NeuronView(NeuronView&&) noexcept = default;
    NeuronView& operator=(NeuronView&&) noexcept = default;

    explicit NeuronView(const NeuronView&) = delete;
    NeuronView& operator=(const NeuronView&) = delete;


    static bool isAtomic() { return false; }

    void clearSpikes() {
        spikes_.clear();
    }

    // Add methods to add callbacks
    template <typename Type>
    void insert(Type& obj, const bool addCallback = true) {
        ViewBase::template insert<Type>(obj);
        if constexpr (requires(Type*, std::function<void(Type*)> f) { static_cast<Type*>(nullptr)->registerPushCallback(f); }) {
            const int id = obj.id();
            if (addCallback) {
                const size_t cbId = obj.registerPushCallback([this](Type* neuron) { onPushActivateCallback(neuron); });
                callbackIds_[id] = cbId;
            }
        }
    }

    template <typename Type>
    void insert(Type* ptr, const bool addCallback = true) {
        ViewBase::template insert<Type>(ptr);
        if constexpr (requires(Type*, std::function<void(Type*)> f) { static_cast<Type*>(nullptr)->registerPushCallback(f); }) {
            const auto id = ptr->id();
            if (ptr && addCallback) {
                const size_t cbId = ptr->registerPushCallback([this](Type* neuron) { onPushActivateCallback(neuron); });
                callbackIds_[id] = cbId;
            }
        }
    }

    template <typename Type, typename... Args>
    Type& emplace(Args&&... args, const bool addCallback = true) {
        Type& obj = this->template emplace<Type>(std::forward<Args>(args)...);
        if constexpr (requires(Type t, std::function<void(Type*)> f) { t.registerPushCallback(f); }) {
            if (addCallback) {
                const auto id = obj.id();
                Type* ptr = &obj;
                const size_t cbId = obj.registerPushCallback([this](Type* neuron) { onPushActivateCallback(neuron); });
                callbackIds_[id] = cbId;
            }
        }
        return obj;
    }

    template <typename Type>
    bool remove(Type& obj) {
        const auto id = obj.id();
        const auto it = callbackIds_.find(id);
        if (it != callbackIds_.end()) {
            if constexpr (requires(Type t, size_t i) { t.unregisterPushCallback(i); }) {
                obj.unregisterPushCallback(it->second);
            }
            callbackIds_.erase(it);
        }
        return this->template remove<Type>(obj);
    }

    template <typename Type>
    bool removeAt(size_t localIndex) {
        auto& vec = get<Type>();
        if (localIndex >= vec.size()) {
            return false;
        }
        int id;
        Type* ptr_to_unregister = nullptr;
        if constexpr (this->PointerStorage) {
            Type* ptr = vec[localIndex];
            if (!ptr) {
                return this->template removeAt<Type>(localIndex);
            }
            id = ptr->id();
            ptr_to_unregister = ptr;
        } else {
            Type& obj = vec[localIndex];
            id = obj.id();
            ptr_to_unregister = &obj;
        }
        const auto it = callbackIds_.find(id);
        if (it != callbackIds_.end()) {
            if constexpr (requires(Type t, size_t i) { t.unregisterPushCallback(i); }) {
                if (ptr_to_unregister) {
                    ptr_to_unregister->unregisterPushCallback(it->second);
                }
            }
            callbackIds_.erase(it);
        }
        return this->removeAt<Type>(localIndex);
    }

    template <typename Type>
    bool swapRemoveAt(size_t localIndex) {
        auto& vec = get<Type>();
        if (localIndex >= vec.size()) {
            return false;
        }
        int id;
        Type* ptr_to_unregister = nullptr;
        if constexpr (this->PointerStorage) {
            Type* ptr = vec[localIndex];
            if (!ptr) {
                return this->swapRemoveAt<Type>(localIndex);
            }
            id = ptr->id();
            ptr_to_unregister = ptr;
        } else {
            Type& obj = vec[localIndex];
            id = obj.id();
            ptr_to_unregister = &obj;
        }
        const auto it = callbackIds_.find(id);
        if (it != callbackIds_.end()) {
            if constexpr (requires(Type t, size_t i) { t.unregisterPushCallback(i); }) {
                if (ptr_to_unregister) {
                    ptr_to_unregister->unregisterPushCallback(it->second);
                }
            }
            callbackIds_.erase(it);
        }
        return this->swapRemoveAt<Type>(localIndex);
    }


    // Visualization methods
    bool isVisualized() const { return isVisualized_; }
    void setVisualized(const bool viz) { isVisualized_ = viz; }
    std::string name() const { return name_; }
    void setName(const std::string& name) { name_ = name; }
    void setVisualLimit(const int limit) { visualLimit_ = limit; }
    int visualLimit() const { return visualLimit_; }
    std::string color() const {
        // Format the string as rgb(r, g, b)
        return "rgb(" + std::to_string(color_[0]) + ", " + std::to_string(color_[1]) + ", " + std::to_string(color_[2]) + ")";
    }
    void setColor(const int r, const int g, const int b) { color_[0] = r, color_[1] = g, color_[2] = b; }
    void setColor(const std::string& color) {
        const std::regex hexRegex("^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$");
        const std::regex rgbRegex(R"(^rgb\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)$)");
        std::smatch match;

        if (std::regex_match(color, match, hexRegex)) {
            // Handle hex color
            std::string hex = match[1].str();

            if (hex.size() == 3) {
                // Convert 3-digit hex to 6-digit hex
                hex = std::string(1, hex[0]) + std::string(1, hex[0]) +
                      std::string(1, hex[1]) + std::string(1, hex[1]) +
                      std::string(1, hex[2]) + std::string(1, hex[2]);
            }

            // Convert hex to RGB values
            color_[0] = std::stoi(hex.substr(0, 2), nullptr, 16);
            color_[1] = std::stoi(hex.substr(2, 2), nullptr, 16);
            color_[2] = std::stoi(hex.substr(4, 2), nullptr, 16);
        } else if (std::regex_match(color, match, rgbRegex)) {
            // Handle rgb(r, g, b) color
            color_[0] = std::stoi(match[1].str());
            color_[1] = std::stoi(match[2].str());
            color_[2] = std::stoi(match[3].str());
        } else {
            throw std::invalid_argument("Invalid color format");
        }
    }

    json serialize() const {
        auto memberArray = json::array();
        this->forEach([&memberArray](auto& neuron) {
            memberArray.push_back(neuron.id());
        });

        auto data = json();
        data["id"] = this->id();
        data["name"] = name();
        data["visualized"] = isVisualized();
        data["color"] = json::array({ color_[0], color_[1], color_[2] });
        data["visual_number"] = visualLimit_;
        data["number"] = this->size();
        data["member_ids"] = memberArray;
        return data;
    }

    std::vector<std::tuple<float, int, int>> visualSpikeTimes() const {
        auto spikeTimes = std::vector<std::tuple<float, int, int>>();
        const float ct = this->currentTime();
        int idx = 0;
        this->forEachWhile([&](auto& neuron) -> bool {
            if (neuron.spike()) { spikeTimes.push_back({ct, idx, this->id()}); }
            ++idx;
            return idx < visualLimit_;
        });
        return spikeTimes;
    }
    std::vector<float> visualSpikes() const {
        auto spikes = std::vector(visualLimit_, 0.f);
        int idx = 0;
        this->forEachWhile([this, &idx, &spikes](auto& neuron) -> bool {
            if (neuron.spike()) { spikes[idx] = this->id(); }
            ++idx;
            return idx < visualLimit_;
        });
        return spikes;
    }
    std::vector<float> visualPotentials() const {
        auto potentials = std::vector<float>();
        int idx = 0;
        this->forEachWhile([this, &idx, &potentials](auto& module) -> bool {
            potentials.push_back(static_cast<float>(module.potential()));
            ++idx;
            return idx < visualLimit_;
        });
        return potentials;
    }

    // Metrics methods
    size_t numOutputs() const {
        size_t count = 0;
        this->forEach([&count](auto& neuron) {
            count += neuron.axon().numOutputs();
        });
        return 0;
    }
    size_t numInputs() const {
        size_t count = 0;
        this->forEach([&count](auto& neuron) {
            count += neuron.soma().numInputs();
            for (auto& dendrite : neuron.dendrites()) {
                count += dendrite.numInputs();
            }
        });
        return 0;
    }
    int numSpikes() const { return spikes_.size(); }
    float activity() const { return spikes_.size() / this->size() * 1000 * this->dt(); }

protected:
    void onPushActivateCallback(auto& module) {
        const auto id = module->id();
        if (const auto tick = this->T(); tick > lastPushUpdate_) {
            spikes_.clear();
            lastPushUpdate_ = tick;
        }
        spikes_.push_back(id);
    }

private:
    void updateIds() {
        ids_.clear();
        ids_.reserve(this->size());
        int i = 0;
        forEach([this, &i](auto& member) {
            ids_[i] = member->id();
            i++;
        });
    }

    // Random generator
    std::mt19937 rng_;

    // Activity members
    std::vector<int> ids_;
    std::vector<int> spikes_;
    std::unordered_map<int, size_t> callbackIds_;
    uint64_t lastPushUpdate_ = 0;

    std::string name_;
    bool isVisualized_ = false;
    int visualLimit_ = 50;
    uint8_t color_[3] = {50, 50, 50};
};

/**
 * A Population is a disjoint group where each Neuron only belongs to exactly one Population.
 */
template <typename MemberTypePack>
class NeuronPopulation final : public Partition<NeuronPopulation<MemberTypePack>, MemberTypePack> {
    friend class Partition<NeuronPopulation, MemberTypePack>;

public:
    struct ConnectomeDistribution {
        std::vector<std::vector<float>> connectome;
        std::map<int, std::vector<int>> distribution;  // conId -> bin counts
        int numBins; float distrMin; float distrMax; float distrStep;
    };
    struct ConnectionClassMeta {
        std::string preName;
        std::string postName;
        std::string conName;
        std::pair<uint64_t, uint64_t> preUuid;
        std::pair<uint64_t, uint64_t> postUuid;
        std::pair<uint64_t, uint64_t> conUuid;
    };

    using PartitionBase = Partition<NeuronPopulation, MemberTypePack>;
    explicit NeuronPopulation(Host* host, const std::string& name = "Neuron Population")
        : Timed(host), PartitionBase(host), name_(name) {}
    ~NeuronPopulation() override = default;

    NeuronPopulation(NeuronPopulation&&) noexcept = default;
    NeuronPopulation& operator=(NeuronPopulation&&) noexcept = default;

    explicit NeuronPopulation(const NeuronPopulation&) = delete;
    NeuronPopulation& operator=(const NeuronPopulation&) = delete;

    static bool isAtomic() { return true; }

    template <typename NeuronType>
    NeuronType& emplace() {
        NeuronType& neuron = static_cast<PartitionBase*>(this)->template emplace<NeuronType>(this->host());
        return neuron;
    }
    template <typename NeuronType, typename... Args>
    NeuronType& emplace(Args&&... args) {
        NeuronType& neuron = static_cast<PartitionBase*>(this)->template emplace<NeuronType>(this->host(), std::forward<Args>(args)...);
        return neuron;
    }

    void update() override {
        PartitionBase::update();
    }
    void clear() {
        PartitionBase::reset();
        PartitionBase::clear();
        neurons_.clear();
        lifMembranes_.clear();
        izhMembranes_.clear();
        connectomeWeights_.clear();
    }

    void clearSpikes() { spikes_.clear(); }   // not needed to clear spikes if membrane is calculated on cpu
    void resetSpikes() {
        clearSpikes();
        this->forEach([](auto& neuron) {
           neuron.resetSpike(); // clears spike in neurons
        });
    }

    // Visualization methods
    std::string name() { return name_; }
    void setName(const std::string& name) { name_ = name; }
    int visualLimit() const { return visualLimit_; }
    void setVisualLimit(const int limit) { visualLimit_ = limit; }
    json serialize() const {
        auto data = json();
        data["id"] = this->id();
        data["name"] = name();
        data["atomic"] = isAtomic();
        data["color"] = json::array({ color_[0], color_[1], color_[2] });
        data["visual_number"] = visualLimit_;
        data["number"] = this->size();
        return data;
    }
    bool callbackMembrane() const { return callback_membrane_; }
    void setCallbackMembrane(const bool active = true) { callback_membrane_ = active; }

    template <typename NeuronType>
    void insertToConnectome(const NeuronType* preNeuron) {
        const int preId = preNeuron->id();
        if constexpr (requires(NeuronType* n) { n->axon(); }) {
            if (preId % 10 == 0) {
                preNeuron->axon().forEachOutput([this, preId](const auto& con) {
                    auto& post = con.post();
                    const Timed* postParent = post.parent();
                    const int postId = postParent->id();
                    if (postId % 10 == 0) {
                        connectomeWeights_[{preId, postId}] = con.weightPtr();
                    }
                });
            }
        }
    }
    ConnectomeDistribution getSubsampleConnectome() const {
        const int n = static_cast<int>(this->size());
        if (n == 0 || connectomeAmount <= 0. || distrStep <= 0.f) {
            return ConnectomeDistribution{};  // Return empty struct
        }

        const int skip = std::max(1, static_cast<int>(1. / connectomeAmount));
        const int m = (n + skip - 1) / skip;
        const int numBins = static_cast<int>(std::ceil((distrMax - distrMin) / distrStep));

        auto connectome = std::vector(m, std::vector(m, 0.0f));
        std::map<int, std::vector<int>> distribution;

        // Initialize distributions for all known connection classes
        for (const auto &conId: connectionClassMeta_ | views::keys) {
            distribution[conId] = std::vector(numBins, 0);
        }

        for (const auto& [pair, weightData] : connectomeWeights_) {
            const auto& [conId, weightPtr] = weightData;
            const int subPre = pair.first;
            const int subPost = pair.second;
            if (subPre < m && subPost < m) {
                const float wgt = *weightPtr;
                connectome[subPre][subPost] += wgt;

                // Compute bin index and clamp to valid range
                int bin = static_cast<int>((wgt - distrMin) / distrStep);
                bin = std::clamp(bin, 0, numBins - 1);
                distribution[conId][bin]++;
            }
        }

        return ConnectomeDistribution{
            std::move(connectome),
            std::move(distribution),
            numBins, distrMin, distrMax, distrStep
        };
    }

    void cacheSubsampleConnectome(const double amount = 0.1) {
        connectomeAmount = amount;
        const int n = static_cast<int>(this->size());
        if (n == 0 || amount <= 0.) { return; }
        const int skip = std::max(1, static_cast<int>(1. / amount));

        std::unordered_map<int, int> map{};
        int count = 0;
        this->forEach([&]<typename NeuronType>(NeuronType& neuron) {
            const int id = neuron.id();
            map[id] = count;
            count++;
        });

        connectomeWeights_.clear();
        connectionClassMeta_.clear();
        connectionClasses_ = 0;

        // Reverse lookup: UUID tuple -> conId
        std::map<std::tuple<std::pair<uint64_t, uint64_t>,
                            std::pair<uint64_t, uint64_t>,
                            std::pair<uint64_t, uint64_t>>, int> classToId;

        this->forEach([&]<typename NeuronType>(NeuronType& preNeuron) {
            const int preId = preNeuron.id();
            if constexpr (requires(NeuronType& n) { n.axon(); n.classId(); }) {
                if (preId % skip == 0) {
                    const int subPre = map[preId] / skip;
                    const auto preClassName = preNeuron.className();
                    const auto preClassUuid = preNeuron.classUuid();

                    preNeuron.axon().forEachOutput([&](const auto& con) {
                        auto& post = con.post();
                        const Timed* postParent = post.parent();
                        const int postId = postParent->id();
                        if (postId % skip == 0) {
                            const int subPost = map[postId] / skip;

                            const auto postClassName = postParent->className();
                            const auto postClassUuid = postParent->classUuid();
                            const auto conClassUuid = con.classUuid();
                            const auto conClassName = con.className(); // assuming this exists

                            // Create lookup key
                            auto key = std::make_tuple(preClassUuid, postClassUuid, conClassUuid);

                            // Find or assign conId
                            int conId;
                            auto it = classToId.find(key);
                            if (it != classToId.end()) {
                                conId = it->second;
                            } else {
                                conId = connectionClasses_++;
                                classToId[key] = conId;

                                // Store metadata
                                connectionClassMeta_[conId] = ConnectionClassMeta{
                                    .preName = preClassName,
                                    .postName = postClassName,
                                    .conName = conClassName,
                                    .preUuid = preClassUuid,
                                    .postUuid = postClassUuid,
                                    .conUuid = conClassUuid
                                };
                            }

                            connectomeWeights_[{subPre, subPost}] = { conId, con.weightPtr() };
                        }
                    });
                }
            }
        });
    }

    std::vector<int> spikes() const { return spikes_; }
    std::vector<float> visualSpikes() {
        std::vector<float> spikes;
        //spikes.assign(visualLimit_, 0);
        int idx = 0;
        forEach([this, &idx, &spikes](auto& module) -> bool {
            spikes.push_back(static_cast<float>(module.spike()));
            ++idx;
            return idx >= visualLimit_;
        });
        return spikes;
    }
    std::vector<float> visualPotentials() const {
        auto potentials = std::vector<float>();
        int idx = 0;
        forEach([this, &idx, &potentials](auto& module) -> bool {
            potentials.push_back(static_cast<float>(module.potential()));
            ++idx;
            return idx >= visualLimit_;
        });
        return potentials;
    }
    std::vector<float> visualCharges() const {
        auto charges = std::vector<float>();
        int idx = 0;
        forEach([this, &idx, &charges](auto& module) -> bool {
            charges.push_back(static_cast<float>(module.charge()));
            ++idx;
            return idx >= visualLimit_;
        });
        return charges;
    }
    std::vector<float> visualInfluences() const {
        auto influences = std::vector<float>();
        int idx = 0;
        forEach([this, &idx, &influences](auto& module) -> bool {
            influences.push_back(static_cast<float>(module.influence()));
            ++idx;
            return idx >= visualLimit_;
        });
        return influences;
    }
    std::vector<float> visualScalings() const {
        auto scalings = std::vector<float>();
        int idx = 0;
        forEach([this, &idx, &scalings](auto& module) -> bool {
            scalings.push_back(static_cast<float>(module.scaling()));
            ++idx;
            return idx >= visualLimit_;
        });
        return scalings;
    }

    // Metrics methods
    int numSpikes() const { return spikes_.size(); }
    float activity() const { return spikes_.size() / this->size() * 1000 * this->dt(); }

protected:
    void onPushActivate(auto& module) { spikes_.push_back(module.id()); }

    void initCuda(const uint64_t custom_seed = static_cast<uint64_t>(-1)) {  // Optional seed
        this->forEach([this](auto& neuron) {
            neurons_.push_back(&neuron);
            if (neuron.template hasMembrane<LifMembrane>()) { lifMembranes_.push_back(&neuron.template membrane<LifMembrane>()); }
            if (neuron.template hasMembrane<IzhMembrane>()) { izhMembranes_.push_back(&neuron.template membrane<IzhMembrane>()); }
        });

        if (!lifMembranes_.empty()) {
            init_cuda_membranes(lifMembranes_, d_I_t, d_S_t, d_V_t, d_U_t, d_T_t, dt_,
                               d_reverse, d_resist, d_V_thresh, d_V_rest,
                               d_adapt_amp, d_refract, d_alpha, d_beta,
                               d_prob, d_rand_states, custom_seed);
        }
        if (!izhMembranes_.empty()) {
            init_cuda_membranes(izhMembranes_, d_I_t, d_S_t, d_V_t, d_U_t, dt_,
                               d_a, d_b, d_c, d_d, d_prob, d_rand_states, custom_seed);
        }
    }

    void updateWithCuda() {
        const size_t n = lifMembranes_.empty() ? izhMembranes_.size() : lifMembranes_.size();

        auto h_I_t = std::vector(n, 0.f);
        auto h_S_t = std::vector<uint8_t>(n, 0u);

        // Clear spikes and withdraw input currents
        clearSpikes();
        int idx = 0;
        this->forEach([&idx, &h_I_t, this](auto& neuron) {
            neuron.resetSpike();
            auto inputCurrent = neuron.soma().withdraw();
            if (neuron.numInputs() > 0) {
                neuron.inputs().forEach([&inputCurrent]<typename InType>(InType& input) {
                    if constexpr (std::is_same_v<typename SignalType_t<InType>::type, decltype(inputCurrent)>) {
                        inputCurrent += input.value();
                    }
                });
            }
            h_I_t[idx] = inputCurrent;
            ++idx;
        });

        if (callback_membrane_) {
            auto h_V_t = std::vector<float>(n);
            if (!lifMembranes_.empty()) {
                update_cuda_lif_membranes(h_I_t, h_S_t, h_V_t);
            }
            if (!izhMembranes_.empty()) {
                update_cuda_izh_membranes(h_I_t, h_S_t, h_V_t);
            }
            idx = 0;
            this->forEach([this, &idx, &h_S_t, &h_V_t](auto& neuron) {
                if (neuron.template hasMembrane<LifMembrane>()) {                   // TODO: this could get runtime constexpr if neuron knows its membrane type!
                    if (idx < static_cast<int>(h_S_t.size()) && h_S_t[idx]) {
                        neuron.activate();
                        this->updateStorage().insert(&neuron);
                    }
                    neuron.template membrane<LifMembrane>().externalUpdate(h_S_t[idx], h_V_t[idx]);
                    ++idx;
                } else if (neuron.template hasMembrane<IzhMembrane>()) {
                    if (idx < static_cast<int>(h_S_t.size()) && h_S_t[idx]) {
                        neuron.activate();
                        this->updateStorage().insert(&neuron);
                    }
                    neuron.template membrane<IzhMembrane>().externalUpdate(h_S_t[idx], h_V_t[idx]);
                    ++idx;
                }
            });
        }
        else {
            if (!lifMembranes_.empty()) {
                update_cuda_lif_membranes(h_I_t, h_S_t);
            }
            if (!izhMembranes_.empty()) {
                update_cuda_izh_membranes(h_I_t, h_S_t);
            }

            idx = 0;
            int spk = 0;
            this->forEach([this, &idx, &h_S_t, &spk](auto& neuron) {
                if (neuron.template hasMembrane<LifMembrane>() || neuron.template hasMembrane<IzhMembrane>()) {                   // TODO: this could get runtime constexpr if neuron knows its membrane type!
                    if (idx < static_cast<int>(h_S_t.size()) && h_S_t[idx]) {
                        neuron.activate();
                        this->updateStorage().insert(&neuron);
                        ++spk;
                    }
                    ++idx;
                }
            });
        }
    }

    void freeCuda() {
        if (!lifMembranes_.empty()) {
            free_device(d_I_t, d_S_t, d_V_t, d_U_t, d_T_t,
                    d_reverse, d_resist, d_V_thresh,
                    d_V_rest, d_adapt_amp, d_refract,
                    d_alpha, d_beta, d_prob, d_rand_states);
            free_streams();

            d_I_t = d_V_t = d_U_t = nullptr;
            d_T_t = d_S_t = nullptr;
            d_reverse = d_resist = d_V_thresh = d_V_rest = nullptr;
            d_adapt_amp = d_alpha = d_beta = d_prob = nullptr;
            d_refract = nullptr;
            d_rand_states = nullptr;
            lifMembranes_.clear();
        }
        if (!izhMembranes_.empty()) {
            free_device(d_I_t, d_S_t, d_V_t, d_U_t,
                        d_thresh, d_a, d_b, d_c, d_d, d_prob, d_rand_states);
            free_streams();

            d_thresh = nullptr;
            d_a = nullptr;
            d_b = nullptr;
            d_c = nullptr;
            d_d = nullptr;
            izhMembranes_.clear();
        }
    }

private:
    std::vector<int> spikes_;

    std::string name_;
    int visualLimit_ = 50;
    uint8_t color_[3] = {50, 50, 50};


    std::vector<Timed*> neurons_;
    std::vector<LifMembrane*> lifMembranes_;
    std::vector<IzhMembrane*> izhMembranes_;
    bool callback_membrane_ = false;

    // Connectome members
    double connectomeAmount = 0.1;
    std::map<std::pair<int, int>, std::tuple<int, const float*>> connectomeWeights_;
    int connectionClasses_ = 0;
    std::map<int, ConnectionClassMeta> connectionClassMeta_;

    const float distrMin = 0.f;
    const float distrMax = 3.5f;
    const float distrStep = 0.025f;

    float dt_;
    float *d_I_t = nullptr, *d_V_t = nullptr, *d_U_t = nullptr;
    uint8_t *d_S_t = nullptr, *d_T_t = nullptr;
    curandState *d_rand_states = nullptr;

    float *d_reverse = nullptr, *d_resist = nullptr;
    float *d_V_thresh = nullptr, *d_V_rest = nullptr, *d_adapt_amp = nullptr;
    float *d_alpha = nullptr, *d_beta = nullptr, *d_prob = nullptr;

    float *d_thresh = nullptr, *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_d = nullptr;

    uint8_t *d_refract = nullptr;
};


#endif // NEURON_COLLECTION_H
