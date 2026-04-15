/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef CALLBACK_H
#define CALLBACK_H

#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>

struct Callback {
    uint64_t tick;
    std::function<void()> action;
    bool operator>(const Callback& other) const { return tick > other.tick; }
};
class CallbackQueue {
public:
    CallbackQueue(float* timestep, uint64_t* tick);
    void schedule(const std::function<void()>& fn, int delay);
    void schedule(const std::function<void()>& fn, const double delayTime) { schedule(fn, static_cast<int>(delayTime / *dt_)); }
    void tick();
private:
    double toTime(const int64_t ticks) const { assert(dt_); return static_cast<double>(ticks) * (*dt_); }
    uint64_t toTick(const double t) const { assert(dt_); return static_cast<uint64_t>(t / (*dt_)); }
    float dt() const { return *dt_; }
    uint64_t T() const { return *tick_; }
    double t() const { return static_cast<double>(*tick_) * (*dt_); }

    float* dt_ = nullptr;
    uint64_t* tick_ = nullptr;
    std::priority_queue<Callback, std::vector<Callback>, std::greater<>> queue;
};

#endif //CALLBACK_H
