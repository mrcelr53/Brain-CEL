/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef INPUT_H
#define INPUT_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <ranges>

#include "Connection.h"

class PostState;


/**
 * Input holds the structure to receive information from other Nodes with Connection instances
 */
template <typename Derived, typename SignalType, typename ConTypePack, typename ProcessTypePack = TypePack<>, bool IsPointerStorage = true>
class Input : public virtual Timed, public Module<Derived, ProcessTypePack>,
              MixedMultiStorage<ConTypePack, IsPointerStorage, false> {

public:
    static constexpr bool is_pointer_storage = IsPointerStorage;

    using Storage = ConTypePack::template Storage<false>;
    using ModuleBase = Module<Derived, ProcessTypePack>;
    using StorageBase = MixedMultiStorage<ConTypePack, IsPointerStorage, false>;

    explicit Input(Host* host) : Timed(host), ModuleBase(host), StorageBase() {}
    ~Input() override = default;

    Input(Input&&) noexcept = default;
    Input& operator=(Input&&) noexcept = default;

    explicit Input(const Input&) = delete;
    Input& operator=(const Input&) = delete;

    // Simulation interface methods
    /// Causes an internal perturbance
    void backfire() { static_cast<Derived*>(this)->perturb(); }

    /// Inject a default value
    void inject() { static_cast<Derived*>(this)->accumulate(); }
    /// Inject a given value
    void inject(const SignalType& signal) { static_cast<Derived*>(this)->accumulate(signal); }

    /// Inject a given value at specific index
    template <typename SignalSubType>
    void inject(int idx, const SignalSubType& signal) { static_cast<Derived*>(this)->accumulate(idx, signal); }

    /// Propagation logic
    void backpropagate() {  if (!isForwardTriggered()) { static_cast<Derived*>(this)->backward(); } }                                     // No explicit backward call for forward lookup learning rules
    /// Propagation logic
    void backpropagate(const SignalType& signal) { if (!isForwardTriggered()) { static_cast<Derived*>(this)->backward(signal); } }       // No explicit backward call for forward lookup learning rules

    void setForwardTriggered(const bool ft) { forwardTriggered_ = ft; }
    bool isForwardTriggered() const { return forwardTriggered_; }

    // Connection methods
    template <typename ConType>
    auto& inputs() { return static_cast<StorageBase*>(this)->template get<ConType>(); }
    template <typename ConType>
    const auto& inputs() const { return static_cast<const StorageBase*>(this)->template get<ConType>(); }
    template <typename ConType>
    bool hasInput() const { return static_cast<StorageBase*>(this)->template has<ConType>(); }
    template <typename ConType>
    bool containsInput(const ConType& input) { return static_cast<StorageBase*>(this)->template contains<ConType>(input); }

    template <typename ConType>
    size_t numReservedInputs() const { return static_cast<const StorageBase*>(this)->template capacity<ConType>(); }
    size_t numReservedInputs() const { return static_cast<const StorageBase*>(this)->capacity(); }

    template <typename ConType>
    void reserveInputs(size_t size) { static_cast<StorageBase*>(this)->template reserve<ConType>(size); }

    template <typename ConType>
    size_t numInputs() const { return static_cast<const StorageBase*>(this)->template size<ConType>(); }
    size_t numInputs() const { return static_cast<const StorageBase*>(this)->size(); }

    template <typename ConType>
    void insertInput(ConType& input) { static_cast<StorageBase*>(this)->template insert<ConType>(input); }
    template <typename ConType, typename Pre, typename... Args>
    ConType& emplaceInput(Pre&& pre, Args&&... args) {
        using PostType = typename ConType::PostTypeBase;
        ConType& con = static_cast<StorageBase*>(this)->template emplace<ConType>(
            this->host(),
            std::forward<Pre>(pre),
            *static_cast<PostType*>(this),
            std::forward<Args>(args)...
        );
        pre.template insertOutput<ConType>(con);   // add the same connection to pre terminal
        return con;
    }
    template <typename ConType, typename... Args>
    ConType& emplaceInput(Args&&... args) { return static_cast<StorageBase*>(this)->template emplace<ConType>(std::forward<Args>(args)...); }
    template <typename ConType>
    void removeInput(ConType& input) { return static_cast<const StorageBase*>(this)->template remove<ConType>(input); }
    template <typename ConType>
    void removeInputAt(size_t index) { return static_cast<const StorageBase*>(this)->template removeAt<ConType>(index); }
    template <typename ConType>
    void reserve(size_t size) { static_cast<StorageBase*>(this)->template reserve<ConType>(size); }

    template <typename ConType>
    void clearInputs() { static_cast<StorageBase*>(this)->template clear<ConType>(); }
    void clearInputs() { static_cast<StorageBase*>(this)->clear(); }

    template <typename F>
    void forEachInput(F&& func) { static_cast<StorageBase*>(this)->forEach(std::forward<F>(func)); }
    template <typename F>
    void forEachInput(F&& func) const { static_cast<const StorageBase*>(this)->forEach(std::forward<F>(func)); }

protected:
    /// Integrates a default signal. Called from outside by inject()
    void accumulate() {}
    /// Integrates a given signal. Called from outside by inject(signal)
    void accumulate(const SignalType&) {}

    /// Integrates a given signal. Called from outside by inject(index, signal)
    template <typename SignalSubType>
    void accumulate(int, const SignalSubType&) {}

    /// Passes a default signal backward. Called from outside by propagate()
    void backward() {
        forEachInput([&](auto& con) {
            con.backpropagate();
        });
    }
    /// Passes a given signal backward. Called from outside by propagate(signal)
    void backward(const SignalType& signal) {
        forEachInput([&](auto& con) {
            con.backpropagate(signal);
        });
    }

    /// Causes an internal perturbance
    void perturb() {
        this->updateProcesses();
    }

    bool forwardTriggered_ = true;
};

/// Copy-constructible specialization.
/// This ensures hashing works.
namespace std {
    template <typename D, typename Sig, typename Con, typename Proc, bool IsLbw>
    struct hash<Input<D, Sig, Con, Proc, IsLbw>> {
        size_t operator()(const Input<D, Sig, Con, Proc, IsLbw>& obj) const noexcept {
            return hash<const void*>{}(static_cast<const void*>(&obj));
        }
    };
}


// Type traits
template <typename Derived, typename SignalType, typename ConTypePack, typename ProcessTypePack, bool IsLbw>
struct SignalType_t<Input<Derived, SignalType, ConTypePack, ProcessTypePack, IsLbw>> {
    using type = SignalType;
};





#endif //INPUT_H



// // Input-dependent states of input connections
// std::unordered_map<uint64_t, PostState*> postState() { return postStates_; }
// template <typename ConType, typename PreType>
// PostState* postState(const Connection<ConType, PreType, Derived, SignalType>* output) { return postState(output->classId()); }
// PostState* postState(const uint64_t outputClass) { if (hasPostState(outputClass)) { return postStates_[outputClass]; }
//     return nullptr; }
// void addPostState(const uint64_t outputClass, PostState* state) { postStates_[outputClass] = state; }
// bool hasPostState(const uint64_t outputClass) const { return postStates_.contains(outputClass); }
//
// // Input-dependent processes of input connections
// void clearPostProcesses() { postProcesses_.clear(); }
// std::unordered_map<uint64_t, Process*> postProcesses() { return postProcesses_; }
// template <typename ConType, typename PreType>
// Process* postProcess(const Connection<ConType, PreType, Derived, SignalType>* input) { return postProcess(input->classId()); }
// Process* postProcess(const uint64_t inputClass) { if (hasPostProcess(inputClass)) { return postProcesses_[inputClass]; }
//     return nullptr; }
// void addPostProcess(const uint64_t inputClass, Process* process) { postProcesses_[inputClass] = process; }
// bool hasPostProcess(const uint64_t inputClass) const { return postProcesses_.contains(inputClass); }
// void updatePostProcess(const uint64_t inputClass) { if (const auto post = postProcess(inputClass)) { post->update(); } }
// void updatePostProcesses() { for (const auto postProcess: postProcesses_ | std::views::values) { postProcess->update(); } }



/*
/**
 * Input holds the structure to receive information from other Nodes with Connection instances.
 * Dynamic input uses slower heap allocations.
 #1#
template <typename Derived, typename SignalType>
class DynamicInput : public virtual Module, public std::enable_shared_from_this<Input<Derived, SignalType>> {
public:
    explicit DynamicInput(Host* host) : Module(host) {}
    ~DynamicInput() override = default;

    /// Inject a default value
    void inject() { static_cast<Derived*>(this)->accumulate(static_cast<SignalType*>(1)); }
    /// Inject a type specified value
    void inject(const SignalType& signal) { static_cast<Derived*>(this)->accumulate(signal); }
    /// Propagation logic
    void backpropagate() { static_cast<Derived*>(this)->backward(); }

    void backfire() { updatePostProcesses(); }

    // Connection methods
    using AnyConnection = std::shared_ptr<ConnectionHandle>;
    std::vector<AnyConnection> inputConnections() const { return inputConnections_; }
    int numInputs() const { return static_cast<int>(inputConnections_.size()); }
    template <typename ConType, typename PreType>
    void addInput(Connection<ConType, PreType, Derived, SignalType>* input) {
        auto wrapper = std::make_shared<ConnectionHandle>(input);
        inputConnections_.push_back(wrapper);
        input->setPost(std::static_pointer_cast<Input>(this->shared_from_this()));
    }


protected:
    /// Integrate an injected signal
    void accumulate(const SignalType& signal) {}
    /// Pass information backward. Called from outside by propagate() faster non-virtual call due to CRTP
    void backward() {
        for (const auto con : inputConnections()) {
            con->backpropagate(static_cast<SignalType>(1));
        }
    }

private:
    std::vector<AnyConnection> inputConnections_;
    std::unordered_map<uint64_t, Process*> postProcesses_;
    std::unordered_map<uint64_t, PostState*> postStates_;
};*/



