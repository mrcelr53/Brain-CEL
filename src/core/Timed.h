/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef TIMED_H
#define TIMED_H

#include <iostream>
#include <cassert>
#include <vector>
#include <queue>
#include <cstdint>

#include "Host.h"
#include "Callback.h"

/**
 * Timed is the base for any instance that can be managed and synchronized by a Host instance
 */
class Timed {
public:
    explicit Timed(Host* host);
    virtual ~Timed() = default;

    Timed(Timed&&) noexcept = default;
    Timed& operator=(Timed&&) noexcept = default;

    explicit Timed(const Timed&) = delete;
    Timed& operator=(const Timed&) = delete;

    virtual void synchronize(SimulationContext* ctx);

    Host* host() const { return host_; }

    int id() const;
    int localId() const;
    void setLocalId(int localId);
    void setLocalId(int localId, int offset);
    void setId(int globalId);
    void setId(int globalId, int offset);
    void setIdOffset(int offset);

    virtual std::string className() const { return className_; }
    void setClassName(std::string className) { className_ = std::move(className); }
    uint64_t classId() const { return classUuidLow_; }
    std::pair<uint64_t, u_int64_t> classUuid() const { return {classUuidHigh_, classUuidLow_}; }
    void setClassUuid(const std::pair<uint64_t, uint64_t>& uuid) { classUuidLow_ = uuid.first; classUuidHigh_ = uuid.second; }
    void setClassUuid(const uint64_t first, const uint64_t second = 0) { classUuidLow_ = first; classUuidHigh_ = second; }

    // Queue
    /// Checks if a callback queue exists.
    bool hasUpdateQueue() const { return updateQueue_ != nullptr; }
    /// Retrieves the callback queue.
    CallbackQueue* updateQueue() const { return updateQueue_; }
    /// Checks if a fan-out queue exists.
    bool hasPushQueue() const { return pushQueue_ != nullptr; }
    /// Retrieves the fan-out queue.
    CallbackQueue* pushQueue() const { return pushQueue_; }
    /// Checks if a fan-out queue exists.
    bool hasPullQueue() const { return pushQueue_ != nullptr; }
    /// Retrieves the fan-out queue.
    CallbackQueue* pullQueue() const { return pushQueue_; }


    /**
     * Schedules a function to be called after a specified number of ticks in the callback queue.
     */
    template<typename F>
    void delayedUpdate(F&& fn, const int delayTick = 1) {
        assert(hasUpdateQueue());
        if (delayTick > 1) {
            updateQueue()->schedule(std::forward<F>(fn), delayTick);
        } else {
            fn();
        }
    }
    /**
     * Schedules a function to be called after a specified duration in milliseconds in the callback queue.
     */
    template<typename F>
    void delayedUpdate(F&& fn, const double delayTime = 1.) {
        assert(hasUpdateQueue());
        if (delayTime > *dt_) {
            updateQueue()->schedule(std::forward<F>(fn), delayTime);
        } else {
            fn();
        }
    }
    /**
     * Schedules a single function to be called after a specified number of ticks in the fanout queue.
     */
    template<typename F>
    void delayedPush(F&& fnFan, const int delayTick = 1) {
        assert(hasPushQueue());
        auto task = std::forward<F>(fnFan);
        if (delayTick > 1) {
            pushQueue()->schedule(std::move(task), delayTick);
        } else {
            task();
        }
    }
    /**
    * Schedules a single function to be called after a specified duration in milliseconds in the fanout queue.
    */
    template<typename F>
    void delayedPush(F&& fnFan, const float delayTime = 1.) {
        assert(hasPushQueue());
        auto task = std::forward<F>(fnFan);
        if (delayTime > *dt_) {
            pushQueue()->schedule(std::move(task), delayTime);
        } else {
            task();
        }
    }
    /**
     * Schedules a fan-out function to be called after a specified number of ticks in the fanout queue.
     */
    template<typename FFan, typename PtrList>
    void delayedPush(FFan&& fnFan, const PtrList& ptrs, const int delayTick = 1) {
        assert(hasPushQueue());
        auto task = [fnFan = std::forward<FFan>(fnFan), ptrs]() {
            for (auto* p : ptrs) { fnFan(p); }
        };
        if (delayTick > 1) {
            pushQueue()->schedule(std::move(task), delayTick);
        } else {
            task();
        }
    }

    /**
     * Schedules a fan-out function to be called after a specified duration in milliseconds in the fanout queue.
     */
    template<typename FFan, typename PtrList>
    void delayedPush(FFan&& fnFan, const PtrList& ptrs, const float delayTime = 1.) {
        assert(hasPushQueue());
        auto task = [fnFan = std::forward<FFan>(fnFan), ptrs]() {
            for (auto* p : ptrs) { fnFan(p); }
        };
        if (delayTime > *dt_) {
            pushQueue()->schedule(std::move(task), delayTime);
        } else {
            task();
        }
    }


    // Time methods
    float timestep() const { return *dt_; }
    uint64_t currentTick() const {return *tick_; }
    double currentTime() const noexcept { assert(tick_ && dt_); return t(); }
    double toTime(const int64_t ticks) const { assert(dt_); return static_cast<double>(ticks) * (*dt_); }
    uint64_t toTick(const double t) const { assert(dt_); return static_cast<uint64_t>(t / (*dt_)); }

    // Parent
    Timed* parent() const { return parent_; }
    void setParent(Timed* parent) { parent_ = parent; }

protected:
    int idOffset() const { return idOffset_; }
    virtual void onIdSet(int globalId, int offset) {}

    float dt() const { return *dt_; }
    uint64_t T() const { return *tick_;}
    double t() const { return static_cast<double>(*tick_) * (*dt_); }

private:
    Host* host_;
    Timed* parent_ = nullptr;

    float* dt_ = nullptr;
    uint64_t* tick_ = nullptr;
    CallbackQueue* updateQueue_ = nullptr;
    CallbackQueue* pushQueue_ = nullptr;
    CallbackQueue* pullQueue_ = nullptr;

    int id_ = -1;
    int idOffset_ = 0;

    std::string className_;
    uint64_t classUuidHigh_ = 0;
    uint64_t classUuidLow_ = 0;
};


#endif //TIMED_H
