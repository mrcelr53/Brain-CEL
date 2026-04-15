/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef PROCESS_H
#define PROCESS_H

#include "Timed.h"
#include "Schema.h"
#include "core/State.h"

/**
 * Process holds the algorithm for a certain way of digesting information over time
 */
template <typename Derived, typename StateType = DefaultState, typename InSignalType = void, typename OutSignalType = void>
class Process : public virtual Timed {
public:
    using InputSignalType = InSignalType;
    using OutputSignalType = OutSignalType;
    using Schema = StateType::Schema;
    using Values = StateType::Values;

    explicit Process(Host* host, Timed* parent = nullptr)
        : Timed(host) { setParent(parent); }
    Process(Host* host, Timed* parent, const StateType& state, const bool setDefault = true)
        : Timed(host) { setParent(parent); setState(state, setDefault); }
    ~Process() override = default;

    Process(Process&&) noexcept = default;
    Process& operator=(Process&&) noexcept = default;

    explicit Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    OutSignalType update() { return static_cast<Derived*>(this)->compute(); }

    template <typename T = InSignalType>
    std::enable_if_t<!std::is_void_v<T>, OutSignalType>
    update(T input) { return static_cast<Derived*>(this)->compute(input); }

    void reset() {
        if (initialState_) { state_ = *initialState_; static_cast<Derived*>(this)->initialize(*initialState_); }
        else { static_cast<Derived*>(this)->initialize(); }
    }

    StateType& state() { return state_; }
    StateType const& state() const { return state_; }
    Schema schema() const { return state_.schema(); }
    Values values() const { return state_.values(); }

    void setState(const StateType& state, const bool setDefault = false) {
        state_ = state;
        if (setDefault) { initialState_ = state; }
        static_cast<Derived*>(this)->initialize(state);
    }

    json jsonSchema() const { return schemaToJson(schema()); }
    json jsonState() const { return applySchema(schema(), values()); }

protected:
    StateType state_{};

    void initialize() {}
    void initialize(const StateType&) {}

    OutSignalType compute() { return OutSignalType{}; }

    template <typename T = InSignalType>
    std::enable_if_t<!std::is_void_v<T>, OutSignalType>
    compute(T) { return OutSignalType{}; }

private:
    std::optional<StateType> initialState_;
};


#endif //PROCESS_H
