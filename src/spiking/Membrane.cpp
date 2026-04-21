/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#include <iostream>
#include <random>

#include "Membrane.h"

#include "Soma.h"
#include "Network.h"

LifMembrane::LifMembrane(Host* host, const float thresh,
                         const float tau, const float rest, const float resist, const float reverse,
                         const float basefire, const float adapt, const float adapt_tau,
                         const float refract, const uint32_t seed) : Timed(host), Membrane(host, seed) {
    auto& st = state_;
    st.V_thresh = thresh;
    st.V_tau = tau;
    st.V_rest = rest;
    st.resist = resist;
    st.reverse = reverse;
    st.basefire = basefire;
    st.adapt_amp = adapt;
    st.adapt_tau = adapt_tau;
    st.refract_ms = refract;
    refreshDeltaTime(true);
}

void LifMembrane::initialize() {
    refreshDeltaTime(true);
    state_.I_t = 0.f;
    state_.U_t = 0.f;
    state_.T_t = 255u;
    state_.V_t = state_.V_rest;
    state_.spike = false;
}
void LifMembrane::initialize(const LifMembraneState& state) {
    refreshDeltaTime(true);
    state_.I_t = 0.f;
    state_.U_t = 0.f;
    state_.T_t = 255u;
    state_.V_t = state.V_rest;
    state_.spike = false;
}

bool LifMembrane::compute() {
    auto& st = state_;
    const float dt = refreshDeltaTime();

    // Load states
    const uint8_t   T_s = st.T_t;
    const float     I_0 = st.I_t / dt;

    // Refractory counter
    if (T_s == 255u)    { st.T_t = st.refract; }
    else                { st.T_t++; }

    // Refractory period
    if (T_s < st.refract) {
        st.V_t = st.V_rest;
        st.I_t = 0.f;
        st.spike = false;
        return false;
    }

    // Trace update
    st.V_t = st.V_t * st.alpha + st.reverse * (1.f - st.alpha) + st.resist * I_0 * (1.f - st.alpha);
    st.I_t = 0.f;
    st.spike = st.V_t >= (st.adapt_amp != 0.f ? st.V_thresh + st.U_t : st.V_thresh);

    // Random firing
    if (!st.spike && st.prob > 0.) {
        thread_local std::uniform_real_distribution<> dist(0.0, 1.0);
        st.spike = dist(st.rng) < st.prob;
    }

    // Final spike decision
    if (st.spike) {
        st.V_t = st.V_rest;
        st.spike = true;
        st.T_t = 0u;
    }

    // Update threshold adaptation
    if (st.adapt_amp != 0.f) {
        const float beta = std::exp(-dt / st.adapt_tau);
        st.U_t = st.U_t * beta + st.adapt_amp * (1.f - beta) * static_cast<float>(st.spike);
    }

    return false;
}

float LifMembrane::refreshDeltaTime(const bool force) {
    const float dt = this->dt();
    if (!force && dt == state_.last_dt) { return dt; }
    state_.last_dt = dt;
    state_.prob = dt * state_.basefire / 1000.f;

    state_.alpha = std::exp(-dt / state_.V_tau);
    state_.beta = std::exp(-dt / state_.adapt_tau);
    state_.refract = static_cast<uint8_t>(state_.refract_ms / dt);
    return dt;
}


IzhMembrane::IzhMembrane(Host* host, float thresh, float recovery_speed, float sensitivity,
                         float potential_reset, float recover_reset, float init_potential,
                         float init_recovery, float basefire, uint32_t seed) : Timed(host), Membrane(host, seed) {
    auto& st = state_;
    st.V_thresh = thresh;
    st.U_timescale = recovery_speed;
    st.U_sensitivity = sensitivity;
    st.V_reset = potential_reset;
    st.U_reset = recover_reset;
    st.basefire = basefire;
    refreshDeltaTime(true);
}

void IzhMembrane::initialize() {
    refreshDeltaTime(true);
    state_.I_t = 0.f;
    state_.U_t = 0.f;
    state_.V_t = state_.V_reset;
    state_.spike = false;
}
void IzhMembrane::initialize(const IzhMembraneState& state) {
    refreshDeltaTime(true);
    state_.I_t = 0.f;
    state_.U_t = 0.f;
    state_.V_t = state.V_reset;
    state_.spike = false;
}

bool IzhMembrane::compute() {
    auto& st = state_;
    const float dt = refreshDeltaTime();

    const float thresh = st.V_thresh;
    const float a = st.U_timescale;
    const float b = st.U_sensitivity;
    const float c = st.V_reset;
    const float d = st.U_reset;

    // Load states
    const float I_0 = st.I_t;
    const float V_0 = st.V_t;
    const float U_0 = st.U_t;

    // Reset spike
    st.spike = false;

    // Update membrane potential (2x for higher resolution)
    const float dtt = dt / 2;
    float V_t = V_0 + dtt * (0.04f * V_0 * V_0 + 5.f * V_0 + 140.f - U_0 + I_0);
    V_t = V_t + dtt * (0.04f * V_t * V_t + 5.f * V_t + 140.f - U_0 + I_0);          // double integration as in paper
    bool spike = V_t >= thresh;

    // Random firing
    if (!spike && st.prob > 0.) {
        thread_local std::uniform_real_distribution<> dist(0.0, 1.0);
        spike = dist(st.rng) < st.prob;
    }

    // Spike condition
    if (spike) {
        st.V_t = c;
        st.U_t = U_0 + d;
        st.spike = true;
    } else {
        st.V_t = V_t;
        st.U_t = U_0 + dt * a * (b * V_t - U_0);     // uses V_t as in paper
    }

    return spike;
}

float IzhMembrane::refreshDeltaTime(const bool force) {
    const float dt = this->dt();
    state_.prob = dt * state_.basefire / 1000.f;
    return dt;
}
