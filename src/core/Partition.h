/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef PARTITION_H
#define PARTITION_H


#include "Container.h"

/**
 * A Partition stores a heterogeneous pool of types passed at runtime.
 * It owns its members. Even if a member is inserted as reference, the member should only belong to a single Partition.
 * A Partition can be used for simulation calls.
 */
template <typename Derived, typename MemberTypePack,
          bool StoresUpdate = true, bool StoresPush = false, bool StoresPull = false,
          bool IsPointerStorage = false, bool HashesLookups = false>
class Partition : public OwnerContainer,
                  public MixedMultiStorage<MemberTypePack, IsPointerStorage, HashesLookups> {

public:
    using Storage = typename MemberTypePack::template Storage<IsPointerStorage>;
    using StorageBase = MixedMultiStorage<MemberTypePack, IsPointerStorage, HashesLookups>;

    explicit Partition(Host* host) : Timed(host), OwnerContainer(host) {}

    Partition(Partition&&) noexcept = default;
    Partition& operator=(Partition&&) noexcept = default;

    explicit Partition(const Partition&) = delete;
    Partition& operator=(const Partition&) = delete;


    // Interface methods
    void reset() override {
        this->forEach([](auto& module) { module.reset(); });
        if (useCuda_ || cudaInitialized_) {
            static_cast<Derived*>(this)->freeCuda();
            cudaInitialized_ = false;
            useCuda_ = false;
        }
    }
    void save() override {
        this->forEach([](auto& module) { module.save(); });
    }
    void load() override {
        this->forEach([](auto& module) { module.load(); });
    }

    void reserveActive() {
        const auto sz = this->size();
        input_.resize(sz);
    }
    void clearActive() {
        std::ranges::fill(input_, 0);
    }
    void setActive(const std::vector<char>& clamped) {
        input_ = clamped;
    }
    void setActive(const std::vector<bool>& clamped) {
        input_.resize(clamped.size());
        for (size_t i = 0; i < clamped.size(); ++i) {
            input_[i] = clamped[i] ? 1 : 0;
        }
    }
    void setActive(const size_t idx, const char value = 1) {
        if (idx < input_.size()) {
            input_[idx] = value;
        }
    }
    void setActive(const std::vector<size_t>& clampedIdx, const char value = 1) {
        for (const size_t idx : clampedIdx) {
            setActive(idx, value);
        }
    }


    void update() override {
        clearUpdateStorage();
        if (useCuda_) { static_cast<Derived*>(this)->updateWithCuda(); }
        else { updateWithCpu(); }
    }
    void push() override {
        clearPushStorage();
        if (updateStorage_.empty()) { return; }
        updateStorage_.forEach([&](auto& module) {
            if (module.push()) {
                if constexpr (StoresPush) { pushStorage_.insert(module); }
                static_cast<Derived*>(this)->onPushActivate(module);
            }
        });
    }
    void pull() override {
        clearPullStorage();
        if (updateStorage_.empty()) { return; }
        updateStorage_.forEach([&](auto& module) {
            if (module.pull()) {
                if constexpr (StoresPull) { pullStorage_.insert(module); }
                static_cast<Derived*>(this)->onPullActivate(module);
            }
        });
    }

    void pushAll() override {
        clearPushStorage();
        this->forEach([&](auto& module) {
            if (module.push()) {
                if constexpr (StoresPush) { pushStorage_.insert(module); }
                static_cast<Derived*>(this)->onPushActivate(module);
            }
        });
    }
    void pullAll() override {
        clearPullStorage();
        this->forEach([&](auto& module) {
            if (module.pull()) {
                if constexpr (StoresPull) { pullStorage_.insert(module); }
                static_cast<Derived*>(this)->onPullActivate(module);
            }
        });
    }

    bool useCuda() const { return useCuda_; }
    bool cudaInitialized() const { return cudaInitialized_; }
    void setCuda(const bool cuda) override {
        if (useCuda_ == cuda) return;
        const size_t n = this->size();
        if ((!useCuda_ || !cudaInitialized_) && cuda) {
            useCuda_ = true;
            static_cast<Derived*>(this)->initCuda();
            cudaInitialized_ = true;
            //std::cout << "CUDA initialized for " << n << " members.\n";
        } else if (useCuda_ && !cuda) {
            static_cast<Derived*>(this)->freeCuda();
            useCuda_ = false;
            cudaInitialized_ = false;
            //std::cout << "CUDA deinitialized.\n";
        }
    }

    template <typename Type>
    void reserveUpdateStorage(const float amount = 0.1f) {
        updateStorage_.template reserve<Type>(static_cast<size_t>(this->size() * amount));
    }
    template <typename Type>
    void reservePushStorage(const float amount = 0.1f) {
        pushStorage_.template reserve<Type>(static_cast<size_t>(this->size() * amount));
    }
    template <typename Type>
    void reservePullStorage(const float amount = 0.1f) {
        pullStorage_.template reserve<Type>(static_cast<size_t>(this->size() * amount));
    }
    void clearUpdateStorage() {
        if constexpr (StoresUpdate) { updateStorage_.clear(); }
    }
    void clearPushStorage() {
        if constexpr (StoresUpdate) { pushStorage_.clear(); }
    }
    void clearPullStorage() {
        if constexpr (StoresUpdate) { pullStorage_.clear(); }
    }

    MixedMultiStorage<MemberTypePack, true>& updateStorage() { return updateStorage_; }
    MixedMultiStorage<MemberTypePack, true>& pushStorage() { return pushStorage_; }
    MixedMultiStorage<MemberTypePack, true>& pullStorage() { return pullStorage_; }

protected:
    void onUpdateActivate(auto& module) {}
    void onPushActivate(auto& module) {}
    void onPullActivate(auto& module) {}


    void updateWithCuda() {
        // update_cuda_neurongroup(neurons_, d_V_t, d_I_t, d_S_t, d_inputs, d_rand_states,
        //                         dt_, reverse, resist, tau, V_thresh, V_rest, adapt, basefire);
    }


    void initCuda() {
        // init_cuda_neurongroup(neurons_, d_V_t, d_I_t, d_S_t, d_inputs, d_rand_states,
        //                       dt_, reverse, resist, tau, V_thresh, V_rest, adapt, basefire);
    }
    void freeCuda() {
        const size_t n = this->size();
        if (n == 0) return;

        // free_cuda_neurongroup(d_V_t, d_I_t, d_S_t, d_inputs, d_rand_states);
        // d_V_t = nullptr;
        // d_I_t = nullptr;
        // d_S_t = nullptr;
        // d_inputs = nullptr;
        // d_rand_states = nullptr;
    }

private:
    std::vector<char> input_{};
    MixedMultiStorage<MemberTypePack, true> updateStorage_{};
    MixedMultiStorage<MemberTypePack, true> pushStorage_{};
    MixedMultiStorage<MemberTypePack, true> pullStorage_{};

    void updateWithCpu() {
        this->forEachEnumerate([&](const size_t idx, auto& module) {
            if (module.update()) {
                if constexpr (StoresUpdate) { updateStorage_.insert(&module); }
                static_cast<Derived*>(this)->onUpdateActivate(module);
            } else if (!input_.empty() && input_[idx]) {
                module.activate();
                if constexpr (StoresUpdate) { updateStorage_.insert(&module); }
                static_cast<Derived*>(this)->onUpdateActivate(module);
                input_[idx] = false;
            }
        });
    }
    bool useCuda_ = false;
    bool cudaInitialized_ = false;
};




#endif // PARTITION_H
