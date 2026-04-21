/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef CONNECTION_H
#define CONNECTION_H

#include <string>

#include "Module.h"
#include "Process.h"
#include "Input.h"
#include "Output.h"

/**
* Connection enables a directional transfer of information between nodes with minimal communication overhead
 */
template <typename Derived, typename PreType, typename PostType, typename ProcessTypePack = TypePack<>,
          bool EnableLearning = false>
class Connection : public Module<Connection<Derived, PreType, PostType, ProcessTypePack, EnableLearning>, ProcessTypePack> {
public:
    using PreSignalType = SignalType_t<PreType>::type;
    using PostSignalType = SignalType_t<PostType>::type;
    using PreTypeBase = PreType;
    using PostTypeBase = PostType;
    static constexpr bool SignalTypesMatch = std::is_same_v<PreSignalType, PostSignalType>;

    Connection(Host* host, PreType& pre, PostType& post)
        : Timed(host), Module<Connection, ProcessTypePack>(host),
          pre_(pre), post_(post), host_(host) {
        if constexpr (EnableLearning) { learningEnabled_ = true; }
    }
    ~Connection() override = default;

    Connection(Connection&&) noexcept = default;
    Connection& operator=(Connection&&) noexcept = default;

    explicit Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    std::string className() const override {
        return pre_.className() + "->" + post_.className();
    }

    /// Propagation call (default)
    void propagate() { static_cast<Derived*>(this)->forward(); }
    /// Propagation call
    void propagate(const PreSignalType& signal) { static_cast<Derived*>(this)->forward(signal); }

    /// Backpropagation call (default)
    void backpropagate() { static_cast<Derived*>(this)->backward(); }
    /// Backpropagation call
    void backpropagate(const PostSignalType& signal) { static_cast<Derived*>(this)->backward(signal); }

    /// Cause an internal change inside the connection
    __attribute__((always_inline)) void learn() {
        if constexpr (EnableLearning) {
            if (learningEnabled_) {
                static_cast<Derived*>(this)->adapt();
            }
        }
    }

    // Pre and Post
    PreType& pre() const { return pre_; }
    PostType& post() const { return post_; }

    // Learning flag
    template <bool EL = EnableLearning, std::enable_if_t<EL, int> = 0>
    void enableLearning(const bool learn) { learningEnabled_ = learn; }
    template <bool EL = EnableLearning, std::enable_if_t<!EL, int> = 0>
    static void enableLearning(bool) {}

    template <bool EL = EnableLearning, std::enable_if_t<EL, int> = 0>
    bool isLearningEnabled() const { return learningEnabled_; }
    template <bool EL = EnableLearning, std::enable_if_t<!EL, int> = 0>
    static bool isLearningEnabled() { return false; }

protected:
    void forward() { post_.inject(); }
    void forward(const PreSignalType& signal) { post_.inject(signal); }
    void backward() { pre_.reject(); }
    void backward(const PostSignalType& signal) { pre_.reject(signal); }

    __attribute__((always_inline)) void adapt() {
        this->updateProcesses();
    }

    PreType& pre_;
    PostType& post_;

private:
    bool learningEnabled_ = false;
    Host* host_;
};




#endif // CONNECTION_H
