/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef HOST_H
#define HOST_H

#include <string>
#include <chrono>
#include <cmath>

#include "Callback.h"


struct SimulationContext {
    float* timestep;            // time delta at which the host ticks
    float* eventstep;           // time delta that for the precision of a single event
    uint64_t* tick;             // number of network ticks (raw time)
    CallbackQueue* updateQueue_;     // callback option
    CallbackQueue* pushQueue_;       // callback option
    CallbackQueue* pullQueue_;       // callback option
};

/**
 * Host stores and manages a variety of Timed instances
 **/
class Host {
public:
    explicit Host(const std::string& name);
    virtual ~Host() = default;

    Host(Host&&) noexcept = default;
    Host& operator=(Host&&) noexcept = default;

    explicit Host(const Host&) = delete;
    Host& operator=(const Host&) = delete;

    SimulationContext* context() { return &context_; }
    const SimulationContext* context() const { return &context_; }
    CallbackQueue* updateQueue() { return &updateQueue_; }
    CallbackQueue* pushQueue() { return &pushQueue_; }
    CallbackQueue* pullQueue() { return &pullQueue_; }

    float timestep() const { return timestep_; }
    void setTimestep(const float timestep) { timestep_ = timestep; }
    float eventstep() const { return eventstep_; }
    void setEventstep(const float eventstep) { eventstep_ = eventstep; }
    double currentTime() const { return tick_ * timestep_; }
    double currentRealtime() const { return realtime_; }
    u_int64_t currentTick() const { return tick_; }
    void setGlobalLearningFactor(const float learningFactor) { learningFactor_ = learningFactor; }
    float globalLearningFactor() const { return learningFactor_; }

    void advance() { tick_++; }
    void advance(double elapsedRealtime, int elapsedTicks = 1);
    void resetTick() { tick_ = 0; }

    int tickDurationUpdateInterval() const { return tickDurationUpdateInterval_; }
    void setTickDurationUpdateInterval(const int interval = 100) { tickDurationUpdateInterval_ = interval; }
    void setName(const std::string& name) { name_ = name; }
    std::string name() const { return name_; }

private:
    // Context elements
    float timestep_ = 1.;
    float eventstep_ = 1.;
    u_int64_t tick_ = 0;           // more than enough for lifetime of network
    CallbackQueue updateQueue_ = CallbackQueue(&timestep_, &tick_);
    CallbackQueue pushQueue_ = CallbackQueue(&timestep_, &tick_);
    CallbackQueue pullQueue_ = CallbackQueue(&timestep_, &tick_);
    SimulationContext context_ = SimulationContext(&timestep_, &eventstep_, &tick_,
                                                   &updateQueue_, &pushQueue_, &pullQueue_);

    // Further info
    double realtime_ = 0.;          // realtime measurement
    int tickDurationUpdateInterval_ = 100;
    double tickDurationAcc_ = 0.;
    double tickDurationAvg_ = 0.;
    double speedup_ = 0.;
    float learningFactor_ = 1.;

    std::string name_;
};


#endif //HOST_H
