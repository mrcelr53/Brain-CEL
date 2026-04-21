/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef MEMBRANE_H
#define MEMBRANE_H

#include <random>
#include <tuple>

#include "../core/Process.h"


template <typename ConTypePack>
class Soma;
class Network;


template <typename Derived, typename DerivedState>
class Membrane : public Process<Derived, DerivedState, float, bool> {
    friend class Process<Derived, DerivedState, float, bool>;

public:
    explicit Membrane(Host* host, const uint32_t seed = 0) : Timed(host), Process<Derived, DerivedState, float, bool>(host) {
        this->reset();
        setSeed(seed);
    }

    Membrane(Membrane&&) noexcept = default;
    Membrane& operator=(Membrane&&) noexcept = default;

    explicit Membrane(const Membrane&) = delete;
    Membrane& operator=(const Membrane&) = delete;

    void setSeed(const uint32_t seed) {
        constexpr std::hash<uint32_t> hasher;     // randomizes sequential seeding
        uint64_t hashed_seed = hasher(seed);
        hashed_seed ^= 0x9e3779b97f4a7c15ULL;
        this->state_.rng.seed(static_cast<uint32_t>(hashed_seed));
    }

    float withdraw() { const float I_t = this->state_.I_t; resetCurrent(); return I_t; }
    float current() const { return this->state_.I_t; }
    void setCurrent(const float current) { this->state_.I_t = current; }
    void addCurrent(const float current) { this->state_.I_t += current; }
    void resetCurrent() { this->state_.I_t = 0.f; }
};


struct LifMembraneState final : ReflectingState<LifMembraneState> {
    float I_t = 0.f;             // Input current
    float V_t = -70.f;           // Potential trace
    float U_t = 0.f;             // Adaptation trace
    uint8_t T_t = 255u;          // Elapsed ticks since last spike

    float V_tau = 20.f;          // Membrane time constant
    float V_thresh = -50.f;      // Voltage threshold for firing
    float V_rest = -70.f;        // Resting potential
    float resist = 10.f;         // Membrane resistance
    float reverse = -60.f;       // Reversal potential
    float basefire = 3.0f;       // Baseline firing rate (Hz)
    float adapt_amp = 0.f;       // Adaptation amplitude
    float adapt_tau = 500.f;
    uint8_t refract = 0u;        // Refractory period (ticks)

    // dt dependent states
    float last_dt = 0.f;
    float refract_ms = 2.f;
    float alpha = 0.f;
    float beta = 0.f;
    float prob = 0.01;

    bool spike = false;
    std::minstd_rand rng;

    using Self = LifMembraneState;
    static constexpr auto descriptions = std::make_tuple(
        Member{"I_t",               &Self::I_t},
        Member{"V_t",               &Self::V_t},
        Member{"U_t",               &Self::U_t},
        Member{"T_t",               &Self::T_t},
        Member{"V_tau",             &Self::V_tau},
        Member{"V_thresh",          &Self::V_thresh},
        Member{"V_rest",            &Self::V_rest},
        Member{"resist",            &Self::resist},
        Member{"reverse",           &Self::reverse},
        Member{"basefire",          &Self::basefire},
        Member{"adapt_amp",         &Self::adapt_amp},
        Member{"refract",           &Self::refract},
        Member{"adapt_tau",         &Self::adapt_tau}
    );

    using Schema = std::tuple<Field<float>, Field<float>, Field<float>, Field<float>,
                              Field<float>, Field<float>, Field<float>, Field<uint8_t>, Field<float>>;
    using Values = std::tuple<float, float, float, float, float, float, float, uint8_t, float>;

    static Schema schema() {
        return {
            Field<float>("V_tau"),
            Field<float>("V_thresh"),
            Field<float>("V_rest"),
            Field<float>("resist"),
            Field<float>("reverse"),
            Field<float>("basefire"),
            Field<float>("adapt_amp"),
            Field<uint8_t>("refract"),
            Field<float>("adapt_tau"),
        };
    }

    Values values() const {
        return {V_tau, V_thresh, V_rest, resist, reverse, basefire, adapt_amp, refract, adapt_tau};
    }
};
class LifMembrane final : public Membrane<LifMembrane, LifMembraneState> {
    friend class Membrane; friend class Process;

public:
    explicit LifMembrane(Host* host,
                      float thresh = 10.,
                      float tau = 20.,
                      float rest = -70,
                      float resist = 10.,
                      float reverse = -60,
                      float basefire = 0.03,
                      float adapt = 1.f,
                      float adapt_tau = 2.f,
                      float refract = 2.f,
                      uint32_t seed = 0);

    LifMembrane(LifMembrane&&) noexcept = default;
    LifMembrane& operator=(LifMembrane&&) noexcept = default;

    explicit LifMembrane(const LifMembrane&) = delete;
    LifMembrane& operator=(const LifMembrane&) = delete;

    bool spike() const { return state_.spike; }
    float potential() const { return state_.V_t; }
    float adaption() const { return state_.U_t; }
    uint8_t refraction() const { return state_.T_t; }

    // Cuda interface
    void externalUpdate(const bool spike) { state_.spike = spike; }
    void externalUpdate(const bool spike, const float potential) { state_.spike = spike; state_.V_t = potential; }

protected:
    float refreshDeltaTime(bool force = false);
    void initialize();
    void initialize(const LifMembraneState& state);

    bool compute();
    bool compute(const float input) { setCurrent(input); return compute(); }
private:
    bool useCuda_ = false;
};

struct IzhMembraneState final : ReflectingState<IzhMembraneState>  {
    float I_t = 0.f;             // Input current
    float V_t = -70.f;           // Potential trace
    float U_t = 0.f;             // Adaptation trace
    uint8_t T_t = 255u;          // Elapsed ticks since last spike

    float V_thresh = -50.f;      // Voltage threshold for resetting
    float U_timescale = 20.f;    // Recovery timescale
    float U_sensitivity = -50.f; // Recovery sensitivity
    float V_reset = -70.f;       // Potential reset value
    float U_reset = 10.f;        // Recovery reset jump
    float basefire = 3.0f;       // Baseline firing rate (Hz)

    // dt dependent states
    float prob = 0.01;

    bool spike = false;
    std::minstd_rand rng;


    using Self = IzhMembraneState;
    static constexpr auto descriptions = std::make_tuple(
        Member{"I_t",               &Self::I_t},
        Member{"V_t",               &Self::V_t},
        Member{"U_t",               &Self::U_t},
        Member{"T_t",               &Self::T_t},
        Member{"V_thresh",          &Self::V_thresh},
        Member{"U_timescale",         &Self::U_timescale},
        Member{"U_sensitivity",     &Self::U_sensitivity},
        Member{"V_reset",           &Self::V_reset},
        Member{"U_reset",           &Self::U_reset},
        Member{"basefire",          &Self::basefire}
    );

    using Schema = std::tuple<Field<float>, Field<float>, Field<float>, Field<float>,
                              Field<float>, Field<float>>;
    using Values = std::tuple<float, float, float, float, float, float>;

    static Schema schema() {
        return {
            Field<float>("V_thresh"),
            Field<float>("U_timescale"),
            Field<float>("U_sensitivity"),
            Field<float>("V_reset"),
            Field<float>("U_reset"),
            Field<float>("basefire")
        };
    }

    Values values() const {
        return {V_thresh, U_timescale, U_sensitivity, V_reset, U_reset, basefire};
    }
};
class IzhMembrane final : public Membrane<IzhMembrane, IzhMembraneState> {
    friend class Process;
public:
    explicit IzhMembrane(Host* host,
                         float thresh = 10.f,
                         float recovery_speed = 0.02f,       // recovery tau
                         float sensitivity = 0.2f,           // recovery sensitivity
                         float potential_reset = -65.f,      // after spike reset of potential
                         float recover_reset = 8.f,          // after spike reset of recovery
                         float init_potential = -60.,
                         float init_recovery = -13.,
                         float basefire = 0.03f,
                         uint32_t seed = 0);

    IzhMembrane(IzhMembrane&&) noexcept = default;
    IzhMembrane& operator=(IzhMembrane&&) noexcept = default;

    explicit IzhMembrane(const IzhMembrane&) = delete;
    IzhMembrane& operator=(const IzhMembrane&) = delete;

    bool spike() const { return state_.spike; }
    float potential() const { return state_.V_t; }
    float recovery() const { return state_.U_t; }

    void externalUpdate(const bool spike) { state_.spike = spike; }
    void externalUpdate(const bool spike, const float potential) { state_.spike = spike; state_.V_t = potential; }

protected:
    float refreshDeltaTime(bool force = false);
    void initialize();
    void initialize(const IzhMembraneState& state);

    bool compute();
    bool compute(const float input) { setCurrent(input); return compute(); }
private:
    bool useCuda_ = false;
};

#endif // MEMBRANE_H
