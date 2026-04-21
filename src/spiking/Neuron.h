/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef SPIKINGNETWORKCPU_NEURON_H
#define SPIKINGNETWORKCPU_NEURON_H

#include <unordered_set>
#include <vector>
#include <random>

#include "core/Input.h"
#include "core/Output.h"

#include "Soma.h"
#include "Axon.h"
#include "Dendrite.h"
#include "Membrane.h"


enum class Transfer { Transmit, Input, Output, Copy, Paste, Unknown };
static std::string toString(const Transfer func) {
    switch (func) {
        case Transfer::Transmit: return "Transmit";
        case Transfer::Input: return "Input";
        case Transfer::Output: return "Output";
        case Transfer::Copy: return "Copy";
        case Transfer::Paste: return "Paste";
        default: return "Unknown";
    }
}
static Transfer fromString(const std::string& transfer) {
    if (transfer == "Transmit") { return Transfer::Transmit; }
    if (transfer == "Input")    { return Transfer::Input; }
    if (transfer == "Output")   { return Transfer::Output; }
    if (transfer == "Copy")     { return Transfer::Copy; }
    if (transfer == "Paste")    { return Transfer::Paste; }
    return Transfer::Unknown;
}


template <typename InConTypePack, typename OutConTypePack, typename InParamPortTypePack, typename OutParamPortTypePack,
          typename NeuronProcessTypePack = TypePack<LifMembrane>>
class Neuron final : public virtual Timed, public Module<Neuron<InConTypePack, OutConTypePack,
                                                                InParamPortTypePack, OutParamPortTypePack,
                                                                NeuronProcessTypePack>, NeuronProcessTypePack> {
    friend class Module<Neuron, NeuronProcessTypePack>;

public:
    using ModuleBase = Module<Neuron, NeuronProcessTypePack>;

    explicit Neuron(Host* host)
        : Timed(host), ModuleBase(host), soma_(host), axon_(host) {}
    explicit Neuron(Host* host, Soma<InConTypePack>&& soma, Axon<OutConTypePack>&& axon)
        : Timed(host), ModuleBase(host), soma_(std::move(soma)), axon_(std::move(axon)) {}
    ~Neuron() override = default;

    Neuron(Neuron&&) noexcept = default;
    Neuron& operator=(Neuron&&) noexcept = default;

    explicit Neuron(const Neuron&) = delete;
    Neuron& operator=(const Neuron&) = delete;

    Network* network() const { return dynamic_cast<Network*>(this->host()); }

    void setClassName(const std::string& name = "Neuron") {
        Timed::setClassName(name);
        soma_.setClassName(Timed::className() + "(Soma)");
        soma_.setId(this->id());
        axon_.setClassName(Timed::className() + "(Axon)");
        axon_.setId(this->id());
    }

    // Membrane interface
    template <typename MembraneType>
    void insertMembrane(MembraneType& membrane) { this->insertProcess(membrane); }
    template <typename MembraneType, typename... Args>
    MembraneType& emplaceMembrane(Args&&... args) { return this->template emplaceProcess<MembraneType>(std::forward<Args>(args)...); }
    template <typename MembraneType>
    MembraneType& membrane() { return this->template process<MembraneType>(); }
    template <typename MembraneType>
    const MembraneType& membrane() const { return this->template process<MembraneType>(); }
    template <typename MembraneType>
    bool hasMembrane() const { return this->template hasProcess<MembraneType>(); }

    // Getters & Setters
    auto& soma() { return soma_; }
    auto& soma() const { return soma_; }
    auto& axon() { return axon_; }
    auto& axon() const { return axon_; }

    template <typename... Args>
    Dendrite<InConTypePack>& emplaceDendrite(Args&&... args) {
        dendrites_.emplace_back(this->host(), std::forward<Args>(args)...);
        auto& dendrite = dendrites_.back();
        dendrite.setClassName(Timed::className() + "(Dendrite)");
        dendrite.setId(this->id());
        return dendrites_.back();
    }
    void reserveDendrites(size_t size) { dendrites_.reserve(size); }
    auto& dendrites() { return dendrites_; }
    auto& dendrites() const { return dendrites_; }
    size_t numDendrites() const { return static_cast<int>(dendrites_.size()); }

    // Inputs
    template <typename ConType>
    size_t numReservedInputs() const { return inputVars_.template capacity<ConType>(); }
    size_t numReservedInputs() const { return inputVars_.capacity(); }
    template <typename ConType>
    void reserveInputs(size_t size) { inputVars_.template reserve<ConType>(size); }
    template <typename Type, typename... Args>
    auto& emplaceInput(Args&&... args) {  // const std::pair<uint64_t, uint64_t>& uuid, const std::string& name
        auto& input = inputVars_.template emplace<Type>(this->host(), std::forward<Args>(args)...);
        input.setClassName(Timed::className() + "(Input)");   // Class specific name
        input.setId(this->id());        // Instance specific id
        // input.setClassUuid(uuid);       // Class specific id
        // input.setName(name);            // Instance specific name
        return input;
    }
    auto& inputs() { return inputVars_; }
    auto& inputs() const { return inputVars_; }
    size_t numInputs() const { return inputVars_.size(); }

    // Outputs
    template <typename ConType>
    size_t numReservedOutputs() const { return outputVars_.template capacity<ConType>(); }
    size_t numReservedOutputs() const { return outputVars_.capacity(); }
    template <typename ConType>
    void reserveOutputs(size_t size) { outputVars_.template reserve<ConType>(size); }
    template <typename Type, typename... Args>
    auto& emplaceOutput(Args&&... args) {  // const std::pair<uint64_t, uint64_t>& uuid, const std::string& name
        auto& output = outputVars_.template emplace<Type>(this->host(), std::forward<Args>(args)...);
        output.setClassName(Timed::className() + "(Output)");   // Class specific name
        output.setId(this->id());   // Instance specific id
        return output;
    }
    auto& outputs() { return outputVars_; }
    auto& outputs() const { return outputVars_; }
    size_t numOutputs() const { return outputVars_.size(); }

    float clampedPoissonRate() const { return poissonRate_; }
    void setClampedPoissonRate(const float fire) { poissonRate_ = fire; }
    void setClampedPotential(const float potential) { clampedPotential_ = potential; }

    bool spike() const { return isSpiking_; }
    void resetSpike() { isSpiking_ = false; }
    float potential() const {
        if (hasMembrane<LifMembrane>()) {
            if (isSpiking_) { return 10.; }
            return membrane<LifMembrane>().potential();
        }
        if (hasMembrane<IzhMembrane>()) {
            if (isSpiking_) { return 30.; }
            return std::min(30.f, membrane<IzhMembrane>().potential());
        }
        return clampedPotential_;
    }
    float firingRate() const { return rate_; }
    void setFiringRateTau(const float tau = 1000.) { const auto dt = this->dt(); rateTau_ = tau; }

    void setAlwaysActive(const bool active) { isAlwaysActive_ = active; }
    bool isAlwaysActive() const { return isAlwaysActive_; }

    void activate() {
        lastSpikeTime_ = this->T();
        isSpiking_ = true;
        axon_.fire();
        soma_.backfire();
        for (auto& dendrite : dendrites()) { dendrite.backfire(); }
    }

protected:
    void initialize() {
        if (hasMembrane<LifMembrane>()) {
            membrane<LifMembrane>().reset(); membrane<LifMembrane>().resetCurrent();
        } else if (hasMembrane<IzhMembrane>()) {
            membrane<IzhMembrane>().reset(); membrane<IzhMembrane>().resetCurrent();
        }

        resetSpike();
        influence_ = 0.;
        scaling_ = 1.;
    }

    bool execute() {
        // Note: this method is CPU only!

        resetSpike();

        // Calculate membrane activity
        if (this->isSimulated() && (hasMembrane<LifMembrane>() || hasMembrane<IzhMembrane>())) {
            auto inputCurrent = soma().withdraw();
            if (numInputs() > 0) {
                inputVars_.forEach([&inputCurrent]<typename InType>(InType& input) {
                    if constexpr (std::is_same_v<typename SignalType_t<InType>::type, decltype(inputCurrent)>) {
                        inputCurrent += input.value();
                    }
                });
            }
            if (hasMembrane<LifMembrane>() && membrane<LifMembrane>().update(inputCurrent)
                || hasMembrane<IzhMembrane>() && membrane<IzhMembrane>().update(inputCurrent)
                || isAlwaysActive()) {
                activate();
                return true;
            }
        }

        else {
            // Simple Poisson firing
            if (const auto firingProbability = this->dt() * (soma().withdraw() + poissonRate_) / 1000.; firingProbability != 0) {
                thread_local std::uniform_real_distribution<> dist(0.0, 1.0);
                if (dist(getTinyRandomGenerator()) < firingProbability || isAlwaysActive()) {
                    activate();
                    return true;
                }
            }
        }
        return false;
    }

    bool propagate() {
        axon_.propagate();
        outputVars_.forEach([](auto& outVar) { outVar.propagate(); });

        if (hasMembrane<LifMembrane>()) {
            const auto bap = membrane<LifMembrane>().potential();
            soma_.backpropagate(bap);
            for (auto& dendrite : dendrites()) { dendrite.backpropagate(bap); }
        } else if (hasMembrane<IzhMembrane>()) {
            const auto bap = membrane<IzhMembrane>().potential();
            soma_.backpropagate(bap);
            for (auto& dendrite : dendrites()) { dendrite.backpropagate(bap); }
        } else {
            soma_.backpropagate();
            for (auto& dendrite : dendrites()) { dendrite.backpropagate(); }
        }
        return true;
    }

private:
    void onIdSet(int globalId, int offset) override {
        axon_.setId(globalId, offset);
        for (auto& dendrite : dendrites()) {
            dendrite.setId(globalId, offset);
        }
        Timed::onIdSet(globalId, offset);
    }

    Soma<InConTypePack> soma_;
    Axon<OutConTypePack> axon_;
    std::vector<Dendrite<InConTypePack>> dendrites_;

    MixedMultiStorage<InParamPortTypePack> inputVars_;
    MixedMultiStorage<OutParamPortTypePack> outputVars_;

    Transfer transfer_ = Transfer::Transmit;

    bool isAlwaysActive_ = false;
    bool isSpiking_ = false;
    uint64_t lastSpikeTime_ = 0;
    bool isScaling_ = true;
    bool useSample_ = false;

    float rateTau_ = 1000.;
    float rate_ = 0;
    float clampedPotential_ = -50.;
    float poissonRate_ = 0.;
    float influence_ = 0.;
    float scaling_ = 1.;
    float scalingAdaption_ = 0.005;
};


/// Copy-constructible specialization.
/// This ensures hashing works.
namespace std {
    template <typename Pre, typename Post, typename ParamPre, typename ParamPost, typename Proc>
    struct hash<Neuron<Pre, Post, ParamPre, ParamPost, Proc>> {
        size_t operator()(const Neuron<Pre, Post, ParamPre, ParamPost, Proc>& obj) const noexcept {
            return hash<const void*>{}(static_cast<const void*>(&obj));
        }
    };
}




#endif //SPIKINGNETWORKCPU_NEURON_H
