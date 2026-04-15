/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef RULE_H
#define RULE_H

#include <fstream>

#include "Process.h"


/**
 * Rule-States stores time-variables that are modified based on the Rule class
 */
template <class Derived>
struct RuleState {
    using PreState = DefaultState;
    using PostState = DefaultState;
    using ConnectionState = DefaultState;
};

/**
 * Rule is a CRTP mixin at the connection point of two nodes.
 * - onPre:            called once per Pre-Node with access to Pre-State
 * - onPost:           called once per Post-Node with access to Post-State
 * - onConnection:     called per connection with access to Connection-State and precalculated Pre- and Post-States
 * @tparam Derived Implementation class of Rule
 */
template <typename Derived, typename InSignalType = void, typename OutSignalType = void>
struct Rule {
    using PreState        = RuleState<Derived>::PreState;
    using PostState       = RuleState<Derived>::PostState;
    using ConnectionState = RuleState<Derived>::ConnectionState;

    template <typename PreType, typename... Args>
    static void onPreInit(PreType& pre, PreState& preState, Args&&... args) {}

    template <typename PostType, typename... Args>
    static void onPostInit(PostType& post, PostState& postState, Args&&... args) {}

    template <typename ConType, typename PreType, typename PostType, typename... Args>
    static void onConnectionInit(ConType& con, ConnectionState& connectionState, Args&&... args) {}


    template <typename PreType>
    static OutSignalType onPre(PreType& pre, PreState& preState)
            requires requires(PreType& p, PreState& s) { { Derived::onPre(p, s) } ->std::same_as<OutSignalType>; } {
        if constexpr (std::is_same_v<InSignalType, void>) {
            return Derived::onPre(pre, preState);
        } else {
            static_assert(false, "onPre without input parameter requires InSignalType to be void"); return {};
        }
    }
    template <typename PreType, typename Input = InSignalType>
    requires (!std::is_void_v<Input> && std::is_same_v<Input, InSignalType>)
    static OutSignalType onPre(Input input, PreType& pre, PreState& preState)
            requires requires(Input i, PreType& p, PreState& s) { { Derived::onPre(i, p, s) } -> std::same_as<OutSignalType>; } {
        return Derived::onPre(input, pre, preState);
    }


    template <typename PostType>
    static OutSignalType onPost(PostType& post, PostState& postState)
            requires requires(PostType& p, PostState& s) { { Derived::onPost(p, s) } -> std::same_as<OutSignalType>; } {
        if constexpr (std::is_same_v<InSignalType, void>) {
            return Derived::onPost(post, postState);
        } else {
            static_assert(false, "onPost without input parameter requires InSignalType to be void"); return {};
        }
    }
    template <typename PostType, typename Input = InSignalType>
    requires (!std::is_void_v<Input> && std::is_same_v<Input, InSignalType>)
    static OutSignalType onPost(Input input, PostType& post, PostState& postState)
            requires requires(Input i, PostType& p, PostState& s) { { Derived::onPost(i, p, s) } -> std::same_as<OutSignalType>; } {
        return Derived::onPost(input, post, postState);
    }

    template <typename PreType, typename PostType, typename ConType>
    OutSignalType onConnection(ConType& con, ConnectionState& conState,
                             PreType& pre, PreState& preState, PostType& post, PostState& postState) const
            requires requires(ConType& cp, ConnectionState& cs, PreType& ap, PreState& as, PostType& bp, PostState& bs) { { Derived::onConnection(cp, cs, ap, as, bp, bs) } -> std::same_as<OutSignalType>; } {
        if constexpr (std::is_same_v<InSignalType, void>) {
            return Derived::onConnection(con, conState, pre, preState, post, postState);
        }
        else {
            static_assert(false, "onConnection without input parameter requires InSignalType to be void"); return {};
        }
    }
    template <typename PreType, typename PostType, typename ConType, typename Input = InSignalType>
    requires (!std::is_void_v<Input> && std::is_same_v<Input, InSignalType>)
    OutSignalType onConnection(Input input, ConType& con, ConnectionState& conState,
                             PreType& pre, PreState& preState, PostType& post, PostState& postState) const
            requires requires(Input i, ConType& cp, ConnectionState& cs, PreType& ap, PreState& as, PostType& bp, PostState& bs) { { Derived::onConnection(i, cp, cs, ap, as, bp, bs) } -> std::same_as<OutSignalType>; } {
        return Derived::onConnection(input, con, conState, pre, preState, post, postState);
    }
};

// Rule type traits
template <typename T>
struct InputSignalType_t { using type = void; };
template <typename Derived, typename InSignalType, typename OutSignalType>
struct InputSignalType_t<Rule<Derived, InSignalType, OutSignalType>> { using type = InSignalType; };

template <typename T>
struct OutputSignalType_t { using type = void; };
template <typename Derived, typename InSignalType, typename OutSignalType>
struct OutputSignalType_t<Rule<Derived, InSignalType, OutSignalType>> { using type = OutSignalType; };



/**
 * Pre-Node rule wrapper (executed if Pre-Node is active)
 */
template <typename RuleType, typename PreType>
class PreRule final : public Process<PreRule<RuleType, PreType>,
                                     typename RuleState<RuleType>::PreState,
                                     typename InputSignalType_t<RuleType>::type,
                                     typename OutputSignalType_t<RuleType>::type> {
    friend class Process<PreRule,
                         typename RuleState<RuleType>::PreState,
                         typename InputSignalType_t<RuleType>::type,
                         typename OutputSignalType_t<RuleType>::type>;
public:
    using InSignalType  = InputSignalType_t<RuleType>::type;
    using OutSignalType = OutputSignalType_t<RuleType>::type;
    using PreState      = RuleState<RuleType>::PreState;
    using ProcessBase = Process<PreRule, PreState, InSignalType, OutSignalType>;

    PreRule(Host* host, PreType& pre) : Timed(host), ProcessBase(host, &pre), pre_(pre) {
        RuleType::template onPreInit<PreType>(pre_, this->state());
    }
    PreRule(Host* host, PreType& pre, const PreState& state) : Timed(host), ProcessBase(host, &pre, state), pre_(pre) {
        RuleType::template onPreInit<PreType>(pre_, this->state());
    }
    template <typename... Args>
    PreRule(Host* host, PreType& pre, Args&&... args)
        : Timed(host), ProcessBase(host, &pre), pre_(pre) {
        RuleType::template onPreInit<PreType>(pre_, this->state(), std::forward<Args>(args)...);
    }

    PreRule(PreRule&&) noexcept = default;
    PreRule& operator=(PreRule&&) noexcept = default;

    explicit PreRule(const PreRule&) = delete;
    PreRule& operator=(const PreRule&) = delete;

    //PreState& preState() { return this->state(); }

protected:
    void initialize() {
        PreState preState{};
        RuleType::template onPreInit<PreType>(preState);
        this->setState(preState);
    }
    void initialize(const State& state) {
        RuleType::template onPreInit<PreType>(this->state());
        ProcessBase::initialize(state);
    }

    OutSignalType compute() {
        return RuleType::template onPre<PreType>(pre_, this->state());
    }

    template <typename T = InSignalType>
    std::enable_if_t<!std::is_void_v<T>, OutSignalType>
    compute(T input) {
        return RuleType::template onPre<PreType>(input, pre_, this->state());
    }

    void reset() {
        ProcessBase::reset();
    }

private:
    PreType& pre_;
};

/**
 * Post-Node rule wrapper (executed if Post-Node is active)
 */
template <typename RuleType, typename PostType>
class PostRule final : public Process<PostRule<RuleType, PostType>,
                                      typename RuleState<RuleType>::PostState,
                                      typename InputSignalType_t<RuleType>::type,
                                      typename OutputSignalType_t<RuleType>::type> {
    friend class Process<PostRule,
                         typename RuleState<RuleType>::PostState,
                         typename InputSignalType_t<RuleType>::type,
                         typename OutputSignalType_t<RuleType>::type>;
public:
    using InSignalType = InputSignalType_t<RuleType>::type;
    using OutSignalType = OutputSignalType_t<RuleType>::type;
    using PostState = RuleState<RuleType>::PostState;
    using ProcessBase = Process<PostRule, PostState, InSignalType, OutSignalType>;

    PostRule(Host* host, PostType& post) : Timed(host), ProcessBase(host, &post), post_(post) {
        RuleType::template onPostInit<PostType>(post_, this->state());
    }
    PostRule(Host* host, PostType& post, const PostState& state) : Timed(host), ProcessBase(host, &post, state), post_(post) {
        RuleType::template onPostInit<PostType>(post_, this->state());
    }
    template <typename... Args>
    PostRule(Host* host, PostType& post, Args&&... args) : Timed(host), ProcessBase(host, &post), post_(post) {
        RuleType::template onPostInit<PostType>(post_, this->state(), std::forward<Args>(args)...);
    }

    PostRule(PostRule&&) noexcept = default;
    PostRule& operator=(PostRule&&) noexcept = default;

    explicit PostRule(const PostRule&) = delete;
    PostRule& operator=(const PostRule&) = delete;

    //PostState& postState() { return this->state(); }

protected:
    void initialize() {
        PostState postState{};
        RuleType::template onPostInit<PostType>(postState);
        this->setState(postState);
    }
    void initialize(const State& state) {
        RuleType::template onPostInit<PostType>(this->state());
        ProcessBase::initialize(state);
    }

    OutSignalType compute() {
        return RuleType::template onPost<PostType>(post_, this->state());
    }

    template <typename T = InSignalType>
    std::enable_if_t<!std::is_void_v<T>, OutSignalType>
    compute(T input) {
        return RuleType::template onPost<PostType>(input, post_, this->state());
    }

    void reset() {
        ProcessBase::reset();
    }

private:
    PostType& post_;
};

/**
 * Per-connection rule wrapper (executed if either site is active)
 */

template <typename RuleType, typename PreType, typename PostType, typename ConType>
class ConnectionRule final : public Process<ConnectionRule<RuleType, PreType, PostType, ConType>,
                                            typename RuleState<RuleType>::ConnectionState,
                                            typename InputSignalType_t<RuleType>::type,
                                            typename OutputSignalType_t<RuleType>::type> {
    friend class Process<ConnectionRule,
                         typename RuleState<RuleType>::ConnectionState,
                         typename InputSignalType_t<RuleType>::type,
                         typename OutputSignalType_t<RuleType>::type>;
public:
    using InSignalType = InputSignalType_t<RuleType>::type;
    using OutSignalType = OutputSignalType_t<RuleType>::type;
    using PreState        = RuleState<RuleType>::PreState;
    using PostState       = RuleState<RuleType>::PostState;
    using ConnectionState = RuleState<RuleType>::ConnectionState;
    using PreRuleType     = PreRule<RuleType, PreType>;
    using PostRuleType    = PostRule<RuleType, PostType>;
    using ProcessBase     = Process<ConnectionRule, ConnectionState, InSignalType, OutSignalType>;

    ConnectionRule(Host* host, ConType& con, PreType& pre, PostType& post,
                   PreRuleType& preRule, PostRuleType& postRule)
            : Timed(host), ProcessBase(host, &con), con_(con), pre_(pre), post_(post), preRule_(preRule), postRule_(postRule) {
        static_assert(std::is_same_v<typename PreRuleType::PreState, PreState>, "PreRule state does not match");
        static_assert(std::is_same_v<typename PostRuleType::PostState, PostState>, "PostRule state does not match");
        RuleType::template onConnectionInit<ConType, OutSignalType, InSignalType>(con_, this->state(), preRule_.state(), postRule_.state());
    }
    ConnectionRule(Host* host, ConType& con, PreType& pre, PostType& post,
                   PreRuleType& preRule, PostRuleType& postRule, const ConnectionState& state)
            : Timed(host), ProcessBase(host, &con, state), con_(con), pre_(pre), post_(post), preRule_(preRule), postRule_(postRule) {
        static_assert(std::is_same_v<typename PreRuleType::PreState, PreState>, "PreRule state does not match");
        static_assert(std::is_same_v<typename PostRuleType::PostState, PostState>, "PostRule state does not match");
        RuleType::template onConnectionInit<ConType, OutSignalType, InSignalType>(con_, this->state(), preRule_.state(), postRule_.state());
    }
    template <typename... Args>
    ConnectionRule(Host* host, ConType& con, PreType& pre, PostType& post,
                      PreRuleType& preRule, PostRuleType& postRule, Args&&... args)
               : Timed(host), ProcessBase(host, &con), con_(con), pre_(pre), post_(post), preRule_(preRule), postRule_(postRule) {
        static_assert(std::is_same_v<typename PreRuleType::PreState, PreState>, "PreRule state does not match");
        static_assert(std::is_same_v<typename PostRuleType::PostState, PostState>, "PostRule state does not match");
        RuleType::template onConnectionInit<ConType, OutSignalType, InSignalType>(con_, this->state(), preRule_.state(), postRule_.state(), std::forward<Args>(args)...);
    }

    ConnectionRule(ConnectionRule&&) noexcept = default;
    ConnectionRule& operator=(ConnectionRule&&) noexcept = default;

    explicit ConnectionRule(const ConnectionRule&) = delete;
    ConnectionRule& operator=(const ConnectionRule&) = delete;

    //PostState& connectionState() { return this->state(); }

protected:
    void initialize() {
        ConnectionState connectionState{};
        RuleType::template onConnectionInit<ConType, OutSignalType, InSignalType>(connectionState);
        this->setState(connectionState);
    }
    void initialize(const State& state) {
        RuleType::template onConnectionInit<ConType, OutSignalType, InSignalType>(this->state());
        ProcessBase::initialize(state);
    }

    OutSignalType compute() {
        // Pre-post coincidence double update catch
        if (!post_.isForwardTriggered()) {  // Check if post is forward triggered
            if (this->T() > lastConnectionUpdate_ || lastConnectionUpdate_ == UINT64_MAX) {
                lastConnectionUpdate_ = this->T();
                if constexpr (std::is_void_v<OutSignalType>) {
                    RuleType::template onConnection(con_, this->state(), pre_, preRule_.state(), post_, postRule_.state());
                } else {
                    return RuleType::template onConnection(con_, this->state(), pre_, preRule_.state(), post_, postRule_.state());
                }
            }
        }
        if constexpr (std::is_void_v<OutSignalType>) {
            RuleType::template onConnection(con_, this->state(), pre_, preRule_.state(), post_, postRule_.state());
        } else {
            return RuleType::template onConnection(con_, this->state(), pre_, preRule_.state(), post_, postRule_.state());
        }
        if constexpr (!std::is_void_v<OutSignalType>) {
            return {};
        }
        return;
    }

    template <typename T = InSignalType>
    std::enable_if_t<!std::is_void_v<T>, OutSignalType> compute(T input) {
        if constexpr (std::is_void_v<OutSignalType>) {
            RuleType::template onConnection(input, con_, this->state(), pre_, preRule_.state(), post_, postRule_.state());
        } else {
            return RuleType::template onConnection(input, con_, this->state(), pre_, preRule_.state(), post_, postRule_.state());
        }
        if constexpr (!std::is_void_v<OutSignalType>) {
            return {};
        }
        return;
    }

    void reset() {
        preRule_.reset();
        postRule_.reset();
        ProcessBase::reset();
    }

private:
    ConType& con_;
    PreType& pre_;
    PostType& post_;
    PreRuleType& preRule_;
    PostRuleType& postRule_;
    uint64_t lastConnectionUpdate_ = UINT64_MAX;
};



#endif //RULE_H
