/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#pragma once

#include <random>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>


inline std::mt19937& getRandomGenerator() {
    thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

inline std::minstd_rand& getTinyRandomGenerator() {
    thread_local std::minstd_rand rng(std::random_device{}());
    return rng;
}

/**
 * Generates num_points evenly spaced values from start to end, inclusive
 */
inline std::vector<double> linspace(double start, const double end, const size_t num_points) {
    std::vector<double> result(num_points);
    if (num_points == 0) return result;
    if (num_points == 1) {
        result[0] = start;
        return result;
    }
    double step = (end - start) / (num_points - 1);
    std::ranges::generate(result, [n = 0, start, step]() mutable { return start + (n++) * step; });
    return result;
}

inline std::vector<double> generateRegularSpiketrain(const double fire_rate,
                                                    const double burst_rate,
                                                    const double burst_phase_amount,
                                                    const int max_burst_number,
                                                    const double dt = 1.0, // dt in ms
                                                    const double duration = -1.0, // duration in ms
                                                    const int number = -1,
                                                    const double offset = 0.0, // offset in ms
                                                    const bool roll = false) {
    const double fire_interval = 1000.0 / fire_rate; // ms
    int total_steps;
    double effective_duration;
    if (number != -1) {
        effective_duration = number * fire_interval;
        total_steps = static_cast<int>(std::ceil(effective_duration / dt));
        effective_duration = total_steps * dt; // Align to grid
    } else if (duration != -1.0) {
        total_steps = static_cast<int>(duration / dt);
        effective_duration = total_steps * dt;
    } else {
        throw std::invalid_argument("Neither number nor duration given");
    }

    const double burst_interval = 1000.0 / burst_rate; // ms
    const double burst_phase = fire_interval * burst_phase_amount;

    auto spikes = std::vector<double>(total_steps, 0.0);

    // Function to place a train starting at a given phase
    auto place_train = [&](double phase) {
        double effective_phase = phase + offset;
        if (roll) {
            effective_phase = std::fmod(effective_phase, effective_duration);
            if (effective_phase < 0) effective_phase += effective_duration;
        }
        double t = effective_phase;
        if (!roll) {
            if (t < 0) {
                double num_intervals = std::ceil(-t / fire_interval);
                t += num_intervals * fire_interval;
            }
            if (t >= effective_duration) return;
        }
        while (true) {
            double wrapped_t = t;
            if (roll) {
                wrapped_t = std::fmod(t, effective_duration);
                if (wrapped_t < 0) wrapped_t += effective_duration;
            }
            if (wrapped_t >= effective_duration) break;
            int index = static_cast<int>(std::floor(wrapped_t / dt));
            if (index >= 0 && index < total_steps) {
                spikes[index] = 1.0;
            }
            t += fire_interval;
            if (!roll && t >= effective_duration) break;
        }
    };

    // Main train (phase 0)
    place_train(0.0);

    // Burst trains
    int i = 0;
    double phase = 0.0;
    while (phase < burst_phase) {
        if (max_burst_number == -1 || i < max_burst_number) {
            place_train(phase);
        }
        phase += burst_interval;
        ++i;
    }

    return spikes;
}
inline std::vector<double> generateRegularSpiketrain(const double fire_rate,
                                                     const double dt = 1.0,
                                                     const int number = -1,
                                                     const double offset = 0.,
                                                     const bool roll = false) {
    return generateRegularSpiketrain(fire_rate, 1.0, 0.25, -1, dt, -1., number, offset, roll);
}
inline std::vector<double> generateRegularSpiketrain(const double fire_rate,
                                                     const double dt = 1.0,
                                                     const double duration = -1.0,
                                                     const double offset = 0.,
                                                     const bool roll = false) {
    return generateRegularSpiketrain(fire_rate, 1.0, 0.25, -1, dt, duration, -1, offset, roll);
}