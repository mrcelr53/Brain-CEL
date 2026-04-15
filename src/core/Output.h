/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef OUTPUT_H
#define OUTPUT_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <ranges>

#include "Connection.h"

class PreState;


/**
 * Output holds the structure to send information to other Nodes with Connection instances
 */
template <typename Derived, typename SignalType, typename ConTypePack, typename ProcessTypePack = TypePack<>, bool IsPointerStorage = false>
class Output : public virtual Timed, public Module<Derived, ProcessTypePack>,
               MixedMultiStorage<ConTypePack, IsPointerStorage, false> {

public:
    static constexpr bool is_pointer_storage = IsPointerStorage;

    using Storage = ConTypePack::template Storage<false>;
    using ModuleBase = Module<Derived, ProcessTypePack>;
    using StorageBase = MixedMultiStorage<ConTypePack, IsPointerStorage, false>;

    explicit Output(Host* host) : Timed(host), ModuleBase(host), StorageBase() {}
    ~Output() override = default;

    Output(Output&&) noexcept = default;
    Output& operator=(Output&&) noexcept = default;

    explicit Output(const Output&) = delete;
    Output& operator=(const Output&) = delete;

    // Simulation interface methods
    /// Causes an internal disturbance
    void fire() { static_cast<Derived*>(this)->stimulate(); }

    /// Causes forward propagation of a default signal
    void propagate() { static_cast<Derived*>(this)->forward(); }
    /// Causes forward propagation of a given signal
    void propagate(const SignalType& signal) { static_cast<Derived*>(this)->forward(signal); }

    /// Inject a default feedback value
    void reject() { static_cast<Derived*>(this)->adapt(); }
    /// Inject a given feedback value
    void reject(const SignalType& signal) { static_cast<Derived*>(this)->adapt(signal); }


    // Connection methods
    template <typename ConType>
    auto& outputs() { return static_cast<StorageBase*>(this)->template get<ConType>(); }
    template <typename ConType>
    const auto& outputs() const { return static_cast<const StorageBase*>(this)->template get<ConType>(); }
    template <typename ConType>
    bool hasOutput() const { return static_cast<StorageBase*>(this)->template has<ConType>(); }
    template <typename ConType>
    bool containsOutput(const ConType& output) { return static_cast<StorageBase*>(this)->template contains<ConType>(output); }

    template <typename ConType>
    size_t numReservedOutputs() const { return static_cast<const StorageBase*>(this)->template capacity<ConType>(); }
    size_t numReservedOutputs() const { return static_cast<const StorageBase*>(this)->capacity(); }

    template <typename ConType>
    void reserveOutputs(size_t size) { static_cast<StorageBase*>(this)->template reserve<ConType>(size); }

    template <typename ConType>
    size_t numOutputs() const { return static_cast<const StorageBase*>(this)->template size<ConType>(); }
    size_t numOutputs() const { return static_cast<const StorageBase*>(this)->size(); }


    template <typename ConType>
    void insertOutput(ConType& output) { static_cast<StorageBase*>(this)->template insert<ConType>(output); }
    template <typename ConType, typename Post, typename... Args>
    ConType& emplaceOutput(Post&& post, Args&&... args) {
        using PreType = typename ConType::PreTypeBase;
        ConType& con = static_cast<StorageBase*>(this)->template emplace<ConType>(
            this->host(),
            *static_cast<PreType*>(this),
            std::forward<Post>(post),
            std::forward<Args>(args)...
        );
        post.template insertInput<ConType>(con);
        return con;
    }
    template <typename ConType, typename... Args>
    ConType& emplaceOutput(Args&&... args) { return static_cast<StorageBase*>(this)->template emplace<ConType>(std::forward<Args>(args)...); }

    template <typename ConType>
    void removeOutput(ConType& output) { return static_cast<const StorageBase*>(this)->template remove<ConType>(output); }
    template <typename ConType>
    void removeOutputAt(size_t index) { return static_cast<const StorageBase*>(this)->template removeAt<ConType>(index); }
    template <typename ConType>
    void reserve(size_t size) { static_cast<StorageBase*>(this)->template reserve<ConType>(size); }

    template <typename ConType>
    void clearOutputs() { static_cast<StorageBase*>(this)->template clear<ConType>(); }
    void clearOutputs() { static_cast<StorageBase*>(this)->clear(); }

    template <typename F>
    void forEachOutput(F&& func) { static_cast<StorageBase*>(this)->forEach(std::forward<F>(func)); }
    template <typename F>
    void forEachOutput(F&& func) const { static_cast<const StorageBase*>(this)->forEach(std::forward<F>(func)); }

protected:
    /// Updates and perturbs internal states. Called from outside by fire()
    void stimulate() {
        this->updateProcesses();
    }

    /// Passes information forward. Called from outside by propagate()
    void forward() {
        static_cast<StorageBase*>(this)->forEach([&](auto& con) {
            con.propagate();
        });
    }
    /// Passes information forward. Called from outside by propagate(signal)
    void forward(const SignalType& signal) {
        this->forEachOutput([&](auto& con) {
            con.propagate(signal);
        });
    }

    /// Integrates a feedback signal. Called from outside by reject()
    void adapt() {}
    /// Integrates a feedback signal. Called from outside by reject(signal)
    void adapt(const SignalType& signal) {}

};


/// Copy-constructible specialization.
/// This ensures hashing works.
namespace std {
    template <typename D, typename Sig, typename Con, typename Proc>
    struct hash<Output<D, Sig, Con, Proc>> {
        size_t operator()(const Output<D, Sig, Con, Proc>& obj) const noexcept {
            return hash<const void*>{}(static_cast<const void*>(&obj));
        }
    };
}


// Type traits
template <typename Derived, typename SignalType, typename ConTypePack, typename ProcessTypePack>
struct SignalType_t<Output<Derived, SignalType, ConTypePack, ProcessTypePack>> {
    using type = SignalType;
};



#endif //OUTPUT_H
