/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef MODULE_H
#define MODULE_H

#include <memory>
#include <optional>
#include <algorithm>

#include "core/Timed.h"
#include "Storage.h"

/**
 * Module is an active unit that processes information over time
 */
template <typename Derived, typename ProcessTypePack = TypePack<>>
class Module : public virtual Timed, MixedStorage<ProcessTypePack> {
public:
    using StorageBase = MixedStorage<ProcessTypePack>;
    using ProcessPack = ProcessTypePack;

    explicit Module(Host* host) : Timed(host), StorageBase() { static_cast<Derived*>(this)->initialize(); }
    ~Module() override = default;

    Module(Module&&) noexcept = default;
    Module& operator=(Module&&) noexcept = default;

    explicit Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    /// Trigger the object
    bool update() { return static_cast<Derived*>(this)->execute(); }
    /// Share information of the object forward
    bool push() {
        for (auto& callback : pushCallbacks_) {
            if (callback) callback(static_cast<Derived*>(this));
        }
        return static_cast<Derived*>(this)->propagate();
    }
    /// Share information of the object backward
    bool pull() {
        for (auto& callback : pullCallbacks_) {
            if (callback) callback(static_cast<Derived*>(this));   // why did this work without static_cast<Derived*>(this) before?
        }
        return static_cast<Derived*>(this)->backpropagate();
    }
    /// Reset the object
    void reset() { static_cast<Derived*>(this)->initialize(); }
    /// Load the object from a serialized state
    void load() { static_cast<Derived*>(this)->deserialize(); }
    /// Reset the object from a serialized state
    void save() { static_cast<Derived*>(this)->serialize(); }

    void activate() {}

    bool isSimulated() const { return simulated_; }
    void setSimulated(const bool simulated) { simulated_ = simulated; }

    // Callback registration for push (added null checks in push/pull for unregistered)
    size_t registerPushCallback(std::function<void(Derived*)> callback) {
        pushCallbacks_.push_back(std::move(callback));
        return pushCallbacks_.size() - 1;
    }
    void unregisterPushCallback(const size_t callbackId) {
        if (callbackId < pushCallbacks_.size()) {
            pushCallbacks_[callbackId] = nullptr;
        }
    }

    // Callback registration for pull
    size_t registerPullCallback(std::function<void()> callback) {
        pullCallbacks_.push_back(std::move(callback));
        return pullCallbacks_.size() - 1;
    }
    void unregisterPullCallback(const size_t callbackId) {
        if (callbackId < pullCallbacks_.size()) {
            pullCallbacks_[callbackId] = nullptr;
        }
    }

    // Process execution (unchanged for multi-type)
    /// Updates a single process without input, returning its output
    template <typename ProcessType>
    ProcessType::OutputType updateProcess() {
        return static_cast<StorageBase*>(this)->template get<ProcessType>().update();
    }

    /// Updates a single process with input, returning its output
    template <typename ProcessType>
    ProcessType::OutputType updateProcess(ProcessType::InputType input) {
        return static_cast<StorageBase*>(this)->template get<ProcessType>().update(input);
    }

    /// Updates all processes without input or output
    void updateProcesses() { forEachProcess([](auto& process) { process.update(); }); }

    /// Updates specified processes without input, returning a tuple of their outputs
    template <typename... ProcessTypes>
    std::tuple<typename ProcessTypes::OutputType...> updateProcesses() {
        return std::make_tuple(static_cast<StorageBase*>(this)->template get<ProcessTypes>().update()...);
    }

    /// Updates specified processes with a tuple of inputs, returning a tuple of their outputs
    template <typename... ProcessTypes>
    std::tuple<typename ProcessTypes::OutputType...> updateProcesses(std::tuple<typename ProcessTypes::InputType...> inputs) {
        return std::apply([this]<typename... PTs>(PTs&&... inps) {
            return std::make_tuple(static_cast<StorageBase*>(this)->template get<PTs>().update(std::forward<PTs>(inps))...);
        }, std::move(inputs));
    }

    /// Resets a single process
    template <typename ProcessType>
    void resetProcess() { static_cast<StorageBase*>(this)->template get<ProcessType>().reset(); }

    /// Resets all processes
    void resetProcesses() { forEachProcess([](auto& process) { process.reset(); }); }

    // Process management (unchanged for multi-type)
    template <typename ProcessType>
    ProcessType& process() { return static_cast<StorageBase*>(this)->template get<ProcessType>(); }
    template <typename ProcessType>
    const ProcessType& process() const { return static_cast<const StorageBase*>(this)->template get<ProcessType>(); }

    template <typename ProcessType>
    bool hasProcess() const { return static_cast<const StorageBase*>(this)->template has<ProcessType>(); }
    template <typename ProcessType>
    bool containsProcess(const ProcessType& process) const { return static_cast<const StorageBase*>(this)->template contains<ProcessType>(process); }

    size_t numProcesses() const { return static_cast<const StorageBase*>(this)->size(); }

    template <typename ProcessType>
    void insertProcess(ProcessType& process) { static_cast<StorageBase*>(this)->template insert<ProcessType>(process); }

    template <typename ProcessType, typename... Args>
    ProcessType& emplaceProcess(Args&&... args) {
        return static_cast<StorageBase*>(this)->template emplace<ProcessType>(this->host(), std::forward<Args>(args)...);
    }

    template <typename ProcessType>
    void removeProcess() { static_cast<StorageBase*>(this)->template remove<ProcessType>(); }

    void clearProcesses() { static_cast<StorageBase*>(this)->clear(); }

    template <typename F>
    void forEachProcess(F&& func) { static_cast<StorageBase*>(this)->forEach(std::forward<F>(func)); }

protected:
    void initialize() {}
    bool execute() { return true; }
    bool propagate() { return false; }
    bool backpropagate() { return false; }
    void serialize() {}
    void deserialize() {}

    bool simulated_ = false;

private:
    std::vector<std::function<void(Derived*)>> pushCallbacks_;
    std::vector<std::function<void(Derived*)>> pullCallbacks_;
};

// Specialization: No Process
template <typename Derived>
class Module<Derived, TypePack<>> : public virtual Timed {
public:
    using ProcessPack = TypePack<>;

    explicit Module(Host* host) : Timed(host) { static_cast<Derived*>(this)->initialize(); }
    ~Module() override = default;

    Module(Module&&) noexcept = default;
    Module& operator=(Module&&) noexcept = default;

    explicit Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    // Core methods unchanged (no storage impact)
    bool update() { return static_cast<Derived*>(this)->execute(); }
    bool push() {
        for (auto& callback : pushCallbacks_) {
            if (callback) callback(static_cast<Derived*>(this));
        }
        return static_cast<Derived*>(this)->propagate();
    }
    bool pull() {
        for (auto& callback : pullCallbacks_) {
            if (callback) callback(static_cast<Derived*>(this));
        }
        return static_cast<Derived*>(this)->backpropagate();
    }
    void reset() { static_cast<Derived*>(this)->initialize(); }
    void load() { static_cast<Derived*>(this)->deserialize(); }
    void save() { static_cast<Derived*>(this)->serialize(); }

    void activate() {}

    bool isSimulated() const { return simulated_; }
    void setSimulated(const bool simulated) { simulated_ = simulated; }

    // Callbacks unchanged
    size_t registerPushCallback(std::function<void(Derived*)> callback) {
        pushCallbacks_.push_back(std::move(callback));
        return pushCallbacks_.size() - 1;
    }
    void unregisterPushCallback(const size_t callbackId) {
        if (callbackId < pushCallbacks_.size()) {
            pushCallbacks_[callbackId] = nullptr;
        }
    }

    size_t registerPullCallback(std::function<void()> callback) {
        pullCallbacks_.push_back(std::move(callback));
        return pullCallbacks_.size() - 1;
    }
    void unregisterPullCallback(const size_t callbackId) {
        if (callbackId < pullCallbacks_.size()) {
            pullCallbacks_[callbackId] = nullptr;
        }
    }

    template <typename ProcessType>
    ProcessType::OutputType updateProcess() { return {}; }
    template <typename ProcessType>
    ProcessType::OutputType updateProcess(ProcessType::InputType) { return {}; }

    void updateProcesses() {}
    template <typename... ProcessTypes>
    std::tuple<typename ProcessTypes::OutputType...> updateProcesses() { return std::make_tuple(); }
    template <typename... ProcessTypes>
    std::tuple<typename ProcessTypes::OutputType...> updateProcesses(std::tuple<typename ProcessTypes::InputType...>) { return std::make_tuple(); }

    template <typename ProcessType>
    void resetProcess() {}
    void resetProcesses() {}

    template <typename ProcessType>
    ProcessType& process() { return *static_cast<ProcessType*>(nullptr); }
    template <typename ProcessType>
    const ProcessType& process() const { return *static_cast<const ProcessType*>(nullptr); }

    template <typename ProcessType>
    bool hasProcess() const { return false; }
    template <typename ProcessType>
    bool containsProcess(const ProcessType&) const { return false; }

    size_t numProcesses() const { return 0; }

    template <typename ProcessType>
    void insertProcess(ProcessType&) { static_assert(false, "Cannot insert into No-Process Module"); }
    template <typename ProcessType, typename... Args>
    ProcessType& emplaceProcess(Args&&...) { static_assert(false, "Cannot emplace into No-Process Module"); }
    template <typename ProcessType>
    void removeProcess() { static_assert(false, "No processes to remove in No-Process Module"); }
    void clearProcesses() {}

    template <typename F>
    void forEachProcess(F&&) {}

protected:
    void initialize() {}
    bool execute() { return true; }
    bool propagate() { return false; }
    bool backpropagate() { return false; }
    void serialize() {}
    void deserialize() {}

    bool simulated_ = false;

private:
    std::vector<std::function<void(Derived*)>> pushCallbacks_;
    std::vector<std::function<void(Derived*)>> pullCallbacks_;
};


// Specialization: Single Process
template <typename Derived, typename SingleProcess>
class Module<Derived, TypePack<SingleProcess>> : public virtual Timed {
public:
    using ProcessPack = TypePack<SingleProcess>;
    using ProcessType = SingleProcess;

    explicit Module(Host* host) : Timed(host) {
        // Conditionally construct process_ with host if possible
        if constexpr (std::is_constructible_v<SingleProcess, Host*>) {
            process_.emplace(host);
        }
        static_cast<Derived*>(this)->initialize();
    }
    ~Module() override = default;

    Module(Module&&) noexcept = default;
    Module& operator=(Module&&) noexcept = default;

    explicit Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    // Core methods unchanged
    bool update() { return static_cast<Derived*>(this)->execute(); }
    bool push() {
        for (auto& callback : pushCallbacks_) {
            if (callback) callback(static_cast<Derived*>(this));
        }
        return static_cast<Derived*>(this)->propagate();
    }
    bool pull() {
        for (auto& callback : pullCallbacks_) {
            if (callback) callback(static_cast<Derived*>(this));
        }
        return static_cast<Derived*>(this)->backpropagate();
    }
    void reset() { static_cast<Derived*>(this)->initialize(); }
    void load() { static_cast<Derived*>(this)->deserialize(); }
    void save() { static_cast<Derived*>(this)->serialize(); }

    void activate() {}

    bool isSimulated() const { return simulated_; }
    void setSimulated(const bool simulated) { simulated_ = simulated; }

    size_t registerPushCallback(std::function<void(Derived*)> callback) {
        pushCallbacks_.push_back(std::move(callback));
        return pushCallbacks_.size() - 1;
    }
    void unregisterPushCallback(const size_t callbackId) {
        if (callbackId < pushCallbacks_.size()) {
            pushCallbacks_[callbackId] = nullptr;
        }
    }

    size_t registerPullCallback(std::function<void(Derived*)> callback) {
        pullCallbacks_.push_back(std::move(callback));
        return pullCallbacks_.size() - 1;
    }
    void unregisterPullCallback(const size_t callbackId) {
        if (callbackId < pullCallbacks_.size()) {
            pullCallbacks_[callbackId] = nullptr;
        }
    }

    template <typename ProcessType>
    ProcessType::OutputType updateProcess() {
        static_assert(std::is_same_v<ProcessType, SingleProcess>, "ProcessType is not supported");
        return process_.value().update();
    }

    template <typename ProcessType>
    ProcessType::OutputType updateProcess(ProcessType::InputType input) {
        static_assert(std::is_same_v<ProcessType, SingleProcess>, "ProcessType is not supported");
        return process_.value().update(input);
    }

    void updateProcesses() { if (process_.has_value()) process_.value().update(); }

    template <typename... ProcessTypes>
    std::tuple<typename ProcessTypes::OutputType...> updateProcesses() {
        static_assert(sizeof...(ProcessTypes) <= 1 && (sizeof...(ProcessTypes) == 0 || std::is_same_v<ProcessTypes..., SingleProcess>), "ProcessType is not supported");
        if constexpr (sizeof...(ProcessTypes) == 0) {
            process_.value().update();
            return std::make_tuple();
        } else {
            return std::make_tuple(process_.value().update());
        }
    }
    template <typename... ProcessTypes>
    std::tuple<typename ProcessTypes::OutputType...> updateProcesses(std::tuple<typename ProcessTypes::InputType...> inputs) {
        static_assert(sizeof...(ProcessTypes) <= 1 && (sizeof...(ProcessTypes) == 0 || std::is_same_v<ProcessTypes..., SingleProcess>), "Only ProcessType is not supported");
        if constexpr (sizeof...(ProcessTypes) == 0) {
            process_.value().update();
            return std::make_tuple();
        } else {
            auto [input] = inputs;
            return std::make_tuple(process_.value().update(input));
        }
    }

    template <typename ProcessType>
    void resetProcess() {
        static_assert(std::is_same_v<ProcessType, SingleProcess>, "ProcessType is not supported");
        process_.value().reset();
    }
    void resetProcesses() { process_.value().reset(); }

    template <typename ProcessType>
    ProcessType& process() {
        static_assert(std::is_same_v<ProcessType, SingleProcess>, "ProcessType is not supported");
        return process_.value();
    }
    template <typename ProcessType>
    const ProcessType& process() const {
        static_assert(std::is_same_v<ProcessType, SingleProcess>, "ProcessType is not supported");
        return process_.value();
    }

    template <typename ProcessType>
    bool hasProcess() const { return process_.has_value(); }
    template <typename ProcessType>
    bool containsProcess(const ProcessType& proc) const { return std::addressof(process_.value()) == std::addressof(proc); }

    size_t numProcesses() const { return process_.has_value() ? 1 : 0; }

    template <typename ProcessType>
    void insertProcess(ProcessType& process) {
        static_assert(std::is_same_v<ProcessType, SingleProcess>, "ProcessType is not supported");
        if constexpr (std::is_same_v<ProcessType, SingleProcess>) {
            // Reset and emplace from moved value
            process_.reset();
            process_.emplace(std::move(process));
        }
    }

    template <typename ProcessType, typename... Args>
    ProcessType& emplaceProcess(Args&&... args) {
        static_assert(std::is_same_v<ProcessType, SingleProcess>, "Only the single ProcessType is supported");
        process_.emplace(this->host(), std::forward<Args>(args)...);
        return process_.value();
    }

    template <typename ProcessType>
    void removeProcess() {
        static_assert(std::is_same_v<ProcessType, SingleProcess>, "ProcessType is not supported");
        process_.reset();
    }

    void clearProcesses() { process_.reset(); }

    template <typename F>
    void forEachProcess(F&& func) {
        if (process_.has_value()) func(process_.value());
    }

protected:
    void initialize() {}
    bool execute() { return true; }
    bool propagate() { return false; }
    bool backpropagate() { return false; }
    void serialize() {}
    void deserialize() {}

    bool simulated_ = false;

private:
    std::optional<SingleProcess> process_{};  // Avoids overhead with MixedStorage
    std::vector<std::function<void(Derived*)>> pushCallbacks_;
    std::vector<std::function<void(Derived*)>> pullCallbacks_;
};



/// Copy-constructible specialization.
/// This ensures hashing works.
namespace std {
    template <typename D, typename Proc>
    struct hash<Module<D, Proc>> {
        size_t operator()(const Module<D, Proc>& obj) const noexcept {
            return hash<const void*>{}(static_cast<const void*>(&obj));
        }
    };
}


// Type traits

template <typename T>
struct ProcessTypePack_t {
    using type = TypePack<>;
};

template <typename Derived, typename ProcessTypePack>
struct ProcessTypePack_t<Module<Derived, ProcessTypePack>> {
    using type = ProcessTypePack;
};

template <typename PackType>
struct ProcessPackRepacker {
    using type = TypePack<>;
};

/// Helper template to extract and repackage process packs
template <typename... MemberTypes>
struct ProcessPackRepacker<TypeVectorPack<MemberTypes...>> {
    using type = TypePack<typename ProcessTypePack_t<MemberTypes>::type...>;
};




#endif //MODULE_H
