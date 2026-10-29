/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef PLASTICITY_H
#define PLASTICITY_H

#include <cmath>

#include "core/Rule.h"
#include "core/Schema.h"

struct SparseSTDP;
template <>
struct RuleState<SparseSTDP> {
    struct PreState final : ReflectingState<PreState> {
        uint64_t T_0 = 0;       // Time of last impulse
        float dT = 0;           // Simulation Timestep

        float jump_amp = 1.f;
        float trace = 0.f;
        float tau = 20.f;
        float inv_tau = 1.0f / 20.f;

        using Self = PreState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "trace", &Self::trace },
            Member{ "tau", &Self::tau },
            Member{"jump_amp", &Self::jump_amp }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
    struct PostState final : ReflectingState<PostState> {
        uint64_t T_0 = 0;       // Time of last impulse
        float dT = 0;           // Simulation Timestep

        float jump_amp = 1.f;
        float trace = 0.f;
        float tau = 20.f;
        float inv_tau = 1.0f / 20.f;

        using Self = PostState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "trace", &Self::trace },
            Member{ "tau", &Self::tau },
            Member{"jump_amp", &Self::jump_amp }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
    struct ConnectionState final : ReflectingState<ConnectionState> {
        uint64_t T_0 = 0;       // Time of last impulse
        float dT = 0;           // Simulation Timestep

        float a_plus = 1.f;
        float a_minus = 1.f;
        float mu_plus = 0.f;
        float mu_minus = 0.f;

        using Self = ConnectionState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "a_plus", &Self::a_plus },
            Member{ "a_minus", &Self::a_minus },
            Member{"mu_plus", &Self::mu_plus },
            Member{"mu_minus", &Self::mu_minus }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
};
/**
 * Typical Spike-Timing Dependent Plasticity (STDP).
 * Less efficient since a forward and backward table lookup is required.
 */
struct SparseSTDP : Rule<SparseSTDP> {
    template <typename OutType, typename... Args>
    static void onPreInit(OutType& pre, PreState& st, Args&&... args) {
        // Set positional args if given
        auto argTuple = std::make_tuple(std::forward<Args>(args)...);
        if constexpr (sizeof...(Args) >= 1) st.jump_amp     = std::get<0>(argTuple);
        if constexpr (sizeof...(Args) >= 2) st.tau          = std::get<1>(argTuple);

        st.dT = pre.timestep();
        st.inv_tau = 1.0f / st.tau;
    }
    template <typename InType, typename... Args>
    static void onPostInit(InType& post, PostState& st, Args&&... args) {
        // Set positional args if given
        auto argTuple = std::make_tuple(std::forward<Args>(args)...);
        if constexpr (sizeof...(Args) >= 1) st.jump_amp     = std::get<0>(argTuple);
        if constexpr (sizeof...(Args) >= 2) st.tau          = std::get<1>(argTuple);

        st.dT = post.timestep();
        st.inv_tau = 1.0f / st.tau;
    }
    template <typename ConType, typename OutType, typename InType, typename... Args>
    static void onConnectionInit(ConType& con, ConnectionState& st, PreState& preSt, PostState& postSt, Args&&... args) {
        // Set positional args if given
        auto argTuple = std::make_tuple(std::forward<Args>(args)...);
        if constexpr (sizeof...(Args) >= 1) st.a_plus      = std::get<0>(argTuple);
        if constexpr (sizeof...(Args) >= 2) st.a_minus     = std::get<1>(argTuple);
        if constexpr (sizeof...(Args) >= 3) st.mu_plus     = std::get<2>(argTuple);
        if constexpr (sizeof...(Args) >= 4) st.mu_minus    = std::get<3>(argTuple);

        st.dT = con.timestep();
    }

    template <typename OutType>
    static void onPre(OutType& pre, PreState& st) {
        const auto T = pre.currentTick();
        const auto dt = static_cast<float>(T - st.T_0) * st.dT;
        st.trace *= expf(-dt * st.inv_tau);         // use inverse to avoid slower division
        st.trace += st.jump_amp;
        st.T_0 = T;
    }

    template <typename InType>
    static void onPost(InType& post, PostState& st) {
        const auto T = post.currentTick();
        const auto dt = static_cast<float>(T - st.T_0) * st.dT;
        st.trace *= expf(-dt * st.inv_tau);       // use inverse to avoid slower division
        st.trace += st.jump_amp;
        st.T_0 = T;
    }

    template <typename ConType, typename OutType, typename InType>
    static void onConnection(ConType& con, ConnectionState& conSt,
                             OutType& pre, PreState& preSt,
                             InType& post, PostState& postSt) {
        const auto T_0 = conSt.T_0;
        const auto T = con.currentTick();
        if (T_0 == T) { return; }     // exit double call - happens if onPre and onPost get called at the same time

        float dw = 0.;
        if (preSt.T_0 == T) {
            // Pre-spike
            const auto dt_post = static_cast<float>(T - postSt.T_0) * postSt.dT;
            const auto post_trace_proj = postSt.trace * expf(-dt_post * postSt.inv_tau);
            dw -= conSt.a_minus * post_trace_proj;
        }
        if (postSt.T_0 == T) {
            // Post-spike
            const auto dt_pre = static_cast<float>(T - preSt.T_0) * preSt.dT;
            const auto pre_trace_proj = preSt.trace * expf(-dt_pre * preSt.inv_tau);       // use inverse to avoid slower division
            dw += conSt.a_plus * pre_trace_proj;
        }
        con.updateWeight(dw);
        conSt.T_0 = T;
    }
};


struct SparseSTDPf;
template <>
struct RuleState<SparseSTDPf> {
    struct PreState final : ReflectingState<PreState> {
        uint64_t T_0 = 0;       // Time of last impulse
        float dT = 0;           // Simulation Timestep

        float jump_amp = 1.f;
        float trace = 0.f;
        float tau = 20.f;
        float inv_tau = 1.0f / 20.f;

        using Self = PreState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "trace", &Self::trace },
            Member{ "tau", &Self::tau },
            Member{"jump_amp", &Self::jump_amp }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
    struct PostState final : ReflectingState<PostState> {
        uint64_t T_0 = 0;               // Time of last impulse
        float dT = 0;                   // Simulation Timestep

        float jump_amp = 1.f;
        float trace = 0.f;
        float tau = 20.f;
        float inv_tau = 1.0f / 20.f;

        static constexpr size_t BUF_SZ = 128;
        std::array<uint64_t, BUF_SZ> spike_buffer{};
        size_t head = 0;

        using Self = PostState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "trace", &Self::trace },
            Member{ "tau", &Self::tau },
            Member{"jump_amp", &Self::jump_amp }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
    struct ConnectionState final : ReflectingState<ConnectionState> {
        uint64_t T_0 = 0;               // Time of last impulse
        float dT = 0;                   // Simulation Timestep

        float a_plus = 1.f;
        float a_minus = 1.f;
        float mu_plus = 0.f;
        float mu_minus = 0.f;
        float w_max = 1.f;

        size_t post_buffer_pos = 0;

        using Self = ConnectionState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "a_plus", &Self::a_plus },
            Member{ "a_minus", &Self::a_minus },
            Member{"mu_plus", &Self::mu_plus },
            Member{"mu_minus", &Self::mu_minus },
            Member{"w_max", &Self::w_max }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
};
/**
 * Efficient forward-table lookup Spike-Timing Dependent Plasticity (STDP)
 */
struct SparseSTDPf : Rule<SparseSTDPf> {
    template <typename OutType, typename... Args>
    static void onPreInit(OutType& pre, PreState& st, Args&&... args) {
        // Set positional args if given
        auto argTuple = std::make_tuple(std::forward<Args>(args)...);
        if constexpr (sizeof...(Args) >= 1) st.jump_amp     = std::get<0>(argTuple);
        if constexpr (sizeof...(Args) >= 2) st.tau          = std::get<1>(argTuple);

        st.dT = pre.timestep();
        st.inv_tau = 1.0f / st.tau;
        st.trace = 0.f;
    }
    template <typename InType, typename... Args>
    static void onPostInit(InType& post, PostState& st, Args&&... args) {
        // Set positional args if given
        auto argTuple = std::make_tuple(std::forward<Args>(args)...);
        if constexpr (sizeof...(Args) >= 1) st.jump_amp     = std::get<0>(argTuple);
        if constexpr (sizeof...(Args) >= 2) st.tau          = std::get<1>(argTuple);

        st.dT = post.timestep();
        st.inv_tau = 1.0f / st.tau;
        st.trace = 0.f;
    }
    template <typename ConType, typename OutType, typename InType, typename... Args>
    static void onConnectionInit(ConType& con, ConnectionState& st, PreState& preSt, PostState& postSt, Args&&... args) {
        // Set positional args if given
        auto argTuple = std::make_tuple(std::forward<Args>(args)...);
        if constexpr (sizeof...(Args) >= 1) st.a_plus      = std::get<0>(argTuple);
        if constexpr (sizeof...(Args) >= 2) st.a_minus     = std::get<1>(argTuple);
        if constexpr (sizeof...(Args) >= 3) st.mu_plus     = std::get<2>(argTuple);
        if constexpr (sizeof...(Args) >= 4) st.mu_minus    = std::get<3>(argTuple);
        if constexpr (sizeof...(Args) >= 5) st.w_max       = std::get<4>(argTuple);

        st.dT = con.timestep();
        st.post_buffer_pos = 0;
    }

    template <typename OutType>
    static void onPre(OutType& pre, PreState& st) {
        const auto T = pre.currentTick();
        const auto dt = static_cast<float>(T - st.T_0) * st.dT;
        st.trace *= expf(-dt * st.inv_tau);      // use inverse to avoid slower division
        st.trace += st.jump_amp;
        st.T_0 = T;
    }

    template <typename InType>
    static void onPost(InType& post, PostState& st) {
        const auto T = post.currentTick();
        const auto dt = static_cast<float>(T - st.T_0) * st.dT;
        const auto decay_exp = -dt * st.inv_tau;  // use inverse to avoid slower division
        st.trace *= expf(decay_exp);
        st.trace += st.jump_amp;
        st.T_0 = T;

        // Push new entry
        const size_t write_idx = st.head % PostState::BUF_SZ;
        st.spike_buffer[write_idx] = T;
        ++st.head;
    }

    template <typename ConType, typename OutType, typename InType>
    static void onConnection(ConType& con, ConnectionState& conSt,
                             OutType& pre, PreState& preSt,
                             InType& post, PostState& postSt) {
        const auto T = post.currentTick();
        const auto w = con.weight();

        float dw = 0.f;

        // Catch up on all missed post-spikes
        const size_t buffer_start = conSt.post_buffer_pos;   //std::max(conSt.post_buffer_pos, postSt.tail);
        const auto pre_trace = preSt.trace;
        const auto dT_pre = conSt.dT;
        const auto inv_tau_pre = preSt.inv_tau;
        const uint64_t T_0 = conSt.T_0;
        for (size_t h = buffer_start; h < postSt.head; ++h) {
            const size_t idx = h % PostState::BUF_SZ;
            const auto T_spk = postSt.spike_buffer[idx];
            const auto dT_pre_part = static_cast<float>(T_spk - T_0) * dT_pre;
            const auto pre_trace_proj = pre_trace * expf(-dT_pre_part * inv_tau_pre);       // use inverse to avoid slower division
            dw += pre_trace_proj;
        }
        dw *= conSt.a_plus;
        dw *= powf(conSt.w_max - w, conSt.mu_plus);
        conSt.post_buffer_pos = postSt.head;

        const auto dt_post = static_cast<float>(T - postSt.T_0) * preSt.dT;
        const auto post_trace_proj = postSt.trace * expf(-dt_post * postSt.inv_tau);   // use inverse to avoid slower division
        dw -= powf(w, conSt.mu_minus) * conSt.a_minus * post_trace_proj;

        con.updateWeight(dw);
        conSt.T_0 = T;
    }
};

struct SparseSTDPi;
template <>
struct RuleState<SparseSTDPi> {
    struct PreState final : ReflectingState<PreState> {
        uint64_t T_0 = 0;       // Time of last impulse
        float dT = 0;           // Simulation Timestep

        float pre_trace = 0.f;
        float tau_pre = 20.f;
        float inv_tau_pre = 1.0f / 20.f;

        using Self = PreState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "pre_trace", &Self::pre_trace },
            Member{ "tau_pre", &Self::tau_pre }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
    struct PostState final : ReflectingState<PostState> {
        uint64_t T_0 = 0;       // Time of last impulse
        float dT = 0;           // Simulation Timestep

        float post_trace = 0.f;
        float tau_post = 20.f;
        float inv_tau_post = 1.0f / 20.f;

        using Self = PostState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "post_trace", &Self::post_trace },
            Member{ "tau_post", &Self::tau_post }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
    struct ConnectionState final : ReflectingState<ConnectionState> {
        uint64_t T_0 = 0;       // Time of last impulse
        float dT = 0;           // Simulation Timestep

        float eligibility = 0.f;
        float tau_eligibility = 40.f;
        float inv_tau_eligibility = 1.0f / 40.f;
        float a_plus = 1.f;
        float a_minus = 1.f;

        using Self = ConnectionState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "eligibility", &Self::eligibility },
            Member{ "tau_eligibility", &Self::tau_eligibility },
            Member{ "a_plus", &Self::a_plus },
            Member{ "a_minus", &Self::a_minus }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
};
/**
 * Typical inhibitory Spike-Timing Dependent Plasticity (iSTDP).
 * Less efficient since a forward and backward table lookup is required.
 */
struct SparseSTDPi : Rule<SparseSTDPi> {
    template <typename OutType, typename... Args>
    static void onPreInit(OutType& pre, PreState& st, Args&&... args) {
        st.dT = pre.timestep();
        st.inv_tau_pre = 1.0f / st.tau_pre;
    }
    template <typename InType, typename... Args>
    static void onPostInit(InType& post, PostState& st, Args&&... args) {
        st.dT = post.timestep();
        st.inv_tau_post = 1.0f / st.tau_post;
    }
    template <typename ConType, typename OutType, typename InType, typename... Args>
    static void onConnectionInit(ConType& con, ConnectionState& st, PreState& preSt, PostState& postSt, Args&&... args) {
        st.dT = con.timestep();
        st.inv_tau_eligibility = 1.0f / st.tau_eligibility;
    }

    template <typename OutType>
    static void onPre(OutType& pre, PreState& st) {
        const auto T = pre.currentTick();
        const auto dt = static_cast<float>(T - st.T_0) * st.dT;
        st.pre_trace *= expf(-dt * st.inv_tau_pre);       // use inverse to avoid slower division
        st.pre_trace += 1.f;
        st.T_0 = T;
    }

    template <typename InType>
    static void onPost(InType& post, PostState& st) {
        const auto T = post.currentTick();
        const auto dt = static_cast<float>(T - st.T_0) * st.dT;
        st.post_trace *= expf(-dt * st.inv_tau_post);       // use inverse to avoid slower division
        st.post_trace += 1.f;
        st.T_0 = T;
    }

    template <typename ConType, typename OutType, typename InType>
    static void onConnection(ConType& con, ConnectionState& conSt,
                             OutType& pre, PreState& preSt,
                             InType& post, PostState& postSt) {
        const auto T_0 = conSt.T_0;
        const auto T = con.currentTick();
        if (T_0 == T) { return; }     // exit double call - happens if pre and post happen at the same time

        const auto dt = static_cast<float>(T - T_0) * conSt.dT;
        conSt.eligibility *= expf(-dt * conSt.inv_tau_eligibility);

        float d_elig = 0.;
        if (preSt.T_0 == T) {
            // Pre-spike
            const auto dt_pre = static_cast<float>(T - preSt.T_0) * preSt.dT;
            preSt.pre_trace *= expf(-dt_pre * preSt.inv_tau_pre);            // use inverse to avoid slower division
            d_elig += conSt.a_plus * preSt.pre_trace;
        }
        if (postSt.T_0 == T) {
            // Post-spike
            const auto dt_post = static_cast<float>(T - postSt.T_0) * postSt.dT;
            postSt.post_trace *= expf(-dt_post * postSt.inv_tau_post);       // use inverse to avoid slower division
            d_elig -= conSt.a_minus * postSt.post_trace;
        }
        conSt.eligibility += d_elig;

        con.updateWeight(conSt.eligibility);
        conSt.T_0 = T;
    }
};


// Swapped trace implementation - but window seems correct...
struct SparseSTDPfi;
template <>
struct RuleState<SparseSTDPfi> {
    struct BufferEntry {
        uint64_t time;
        float trace;
    };

    struct PreState final : ReflectingState<PreState> {
        uint64_t T_0 = 0;       // Time of last impulse
        float dT = 0;           // Simulation Timestep

        float pre_trace = 0.f;
        float tau_pre = 20.f;
        float inv_tau_pre = 1.0f / 20.f;

        using Self = PreState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "pre_trace", &Self::pre_trace },
            Member{ "tau_pre", &Self::tau_pre }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
    struct PostState final : ReflectingState<PostState> {
        uint64_t T_0 = 0;               // Time of last impulse
        float dT = 0;                   // Simulation Timestep

        float post_trace = 0.f;
        float tau_post = 20.f;
        float inv_tau_post = 1.0f / 20.f;

        static constexpr size_t BUF_SZ = 128;
        std::array<BufferEntry, BUF_SZ> spike_buffer{};
        size_t head = 0;
        size_t tail = 0;

        using Self = PostState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "post_trace", &Self::post_trace },
            Member{ "tau_post", &Self::tau_post }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
    struct ConnectionState final : ReflectingState<ConnectionState> {
        uint64_t T_0 = 0;               // Time of last impulse
        float dT = 0;                   // Simulation Timestep

        size_t post_buffer_pos = 0;

        float eligibility = 0.f;
        float tau_eligibility = 40.f;
        float inv_tau_eligibility = 1.0f / 40.f;
        float a_plus = 1.f;
        float a_minus = 1.f;

        using Self = ConnectionState;
        static constexpr auto descriptions = std::make_tuple(
            Member{ "eligibility", &Self::eligibility },
            Member{ "tau_eligibility", &Self::tau_eligibility },
            Member{ "a_plus", &Self::a_plus },
            Member{ "a_minus", &Self::a_minus }
        );
        using Schema = decltype(schema());
        using Values = decltype(std::declval<Self>().values());
    };
};
/**
 * Efficient forward-triggered inhibitory Spike-Timing Dependent Plasticity (FT-iSTDP)
 */
struct SparseSTDPfi : Rule<SparseSTDPfi> {
    template <typename OutType, typename... Args>
    static void onPreInit(OutType& pre, PreState& st, Args&&... args) {
        st.dT = pre.timestep();
        st.inv_tau_pre = 1.0f / st.tau_pre;
    }
    template <typename InType, typename... Args>
    static void onPostInit(InType& post, PostState& st, Args&&... args) {
        st.dT = post.timestep();
        st.inv_tau_post = 1.0f / st.tau_post;
    }
    template <typename ConType, typename OutType, typename InType, typename... Args>
    static void onConnectionInit(ConType& con, ConnectionState& st, PreState& preSt, PostState& postSt, Args&&... args) {
        st.dT = con.timestep();
        st.inv_tau_eligibility = 1.0f / st.tau_eligibility;
    }

    template <typename OutType>
    static void onPre(OutType& pre, PreState& st) {
        const auto T = pre.currentTick();
        const auto dt = static_cast<float>(T - st.T_0) * st.dT;
        const auto decay_exp = -dt * st.inv_tau_pre;  // use inverse to avoid slower division
        st.pre_trace *= expf(decay_exp);
        st.pre_trace += 1.f;
        st.T_0 = T;
    }

    template <typename InType>
    static void onPost(InType& post, PostState& st) {
        const auto T = post.currentTick();
        const auto dt = static_cast<float>(T - st.T_0) * st.dT;
        const auto decay_exp = -dt * st.inv_tau_post;  // use inverse to avoid slower division
        st.post_trace *= expf(decay_exp);
        st.post_trace += 1.f;
        st.T_0 = T;

        // //Pop front entries if older than threshold
        // while (!st.spike_buffer.empty()) {
        //     const auto& T_spk = st.spike_buffer.front().first;
        //     const auto front_dt = static_cast<float>(T - T_spk) * st.dT;
        //     if (front_dt > 100.f)      { st.spike_buffer.pop_front(); }
        //     else                        { break; }
        // }
        // st.spike_buffer.push_back({T, st.post_trace});

        // Pop front entries if older than threshold
        while (st.tail < st.head) {
            const size_t front_idx = st.tail % PostState::BUF_SZ;
            const uint64_t T_spk = st.spike_buffer[front_idx].time;
            const auto front_dt = static_cast<float>(T - T_spk) * st.dT;
            if (front_dt > 100.f) {
                ++st.tail;
            } else {
                break;
            }
        }

        // Push new entry
        const size_t write_idx = st.head % PostState::BUF_SZ;
        st.spike_buffer[write_idx] = {T, st.post_trace};
        ++st.head;

        // const auto size = st.head - st.tail;
        // std::cout << "T=" << T << " head=" << st.head << " tail=" << st.tail << " size=" << size << std::endl;
        // for (size_t i = st.head-std::min(size_t{5}, size); i < st.head; ++i) {
        //     const size_t idx = i % PostState::BUF_SZ;
        //     std::cout << "  " << i << ": time=" << st.spike_buffer[idx].time << " trace=" << st.spike_buffer[idx].trace << std::endl;
        // }
        // if (size == 0) { std::cerr << "  WARNING: Spike Buffer empty after push!" << std::endl; }
    }

    template <typename ConType, typename OutType, typename InType>
    static void onConnection(ConType& con, ConnectionState& conSt,
                             OutType& pre, PreState& preSt,
                             InType& post, PostState& postSt) {
        const auto T = post.currentTick();

        float dw = 0.f;
        const size_t buffer_start = std::max(conSt.post_buffer_pos, postSt.tail);
        uint64_t T_0_part = conSt.T_0;
        for (size_t h = buffer_start; h < postSt.head; ++h) {
            const size_t idx = h % PostState::BUF_SZ;
            const auto& entry = postSt.spike_buffer[idx];
            const auto spike_time = entry.time;
            const auto post_trace = entry.trace;
            const auto dT_part = static_cast<float>(spike_time - T_0_part) * conSt.dT;
            conSt.eligibility *= expf(-dT_part * conSt.inv_tau_eligibility);
            conSt.eligibility -= conSt.a_minus * post_trace;
            dw += conSt.eligibility;
            T_0_part = spike_time;
        }
        conSt.post_buffer_pos = postSt.head;

        const auto dt = static_cast<float>(T - T_0_part) * conSt.dT;
        conSt.eligibility *= expf(-dt * conSt.inv_tau_eligibility);
        conSt.eligibility += conSt.a_plus * preSt.pre_trace;
        con.updateWeight(dw + conSt.eligibility);
        conSt.T_0 = T;
    }
};


#endif //PLASTICITY_H
