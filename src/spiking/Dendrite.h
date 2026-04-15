/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef DENDRITE_H
#define DENDRITE_H

#include "core/meta/rule_expander.h"
#include "../core/Input.h"

#include "Synapse.h"

class Network;


template <typename ConTypePack>
class Dendrite final : public virtual Timed, public Input<Dendrite<ConTypePack>, float, ConTypePack, typename PostRulePacker<typename ConTypePack::BaseProcessTypes, Dendrite<ConTypePack>>::type> {
    friend class Input<Dendrite, float, ConTypePack, typename PostRulePacker<typename ConTypePack::BaseProcessTypes, Dendrite>::type>;

public:
    explicit Dendrite(Host* host)
        : Timed(host), Input<Dendrite, float, ConTypePack, typename PostRulePacker<typename ConTypePack::BaseProcessTypes, Dendrite>::type>(host) {}

    Dendrite(Dendrite&&) noexcept = default;
    Dendrite& operator=(Dendrite&&) noexcept = default;

    explicit Dendrite(const Dendrite&) = delete;
    Dendrite& operator=(const Dendrite&) = delete;

    Network* network() const { return dynamic_cast<Network*>(this->host()); }

    float withdraw() { const float I_t = current_; resetCurrent(); return I_t; }
    float current() const { return current_; }
    void setCurrent(const float current) { current_ = current; }
    void addCurrent(const float input) { current_ += input; }
    void resetCurrent() { current_ = 0.f; }

    float bap() const { return bap_; }

protected:
    void accumulate() { current_ += 1; }
    void accumulate(const float value) { current_ += value; }

    void backward() {
        //delayedCall([this] { transmit(); }, delay_);
        this->forEachInput([&](auto& con) {
            con.learn();
            //con.backpropagate();
        });
        lastTransmit_ = this->T();
    }
    void backward(const float signal) {
        bap_ = signal;
        //delayedCall([this] { transmit(); }, delay_);
        this->forEachInput([&](auto& con) {
            con.learn();
            //con.backpropagate(signal);
        });
        lastTransmit_ = this->T();
    }

private:
    float current_ = 0.f;
    float bap_ = 0.f;
    uint64_t lastTransmit_ = 0;
};



// Type traits
template <typename ConTypePack>
struct SignalType_t<Dendrite<ConTypePack>> {
    using type = typename SignalType_t<Input<Dendrite<ConTypePack>, float, ConTypePack,
                                             typename ProcessPackRepacker<ConTypePack>::type>>::type;
};


#endif //DENDRITE_H
