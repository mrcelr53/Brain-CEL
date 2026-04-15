/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef AXON_H
#define AXON_H

#include "core/meta/rule_expander.h"
#include "../core/Output.h"

#include "Synapse.h"


class Network;


enum Transmitter { Glutamate, GABA, UnknownTransmitter, NoTransmitter };


template <typename ConTypePack>
class Axon final : public virtual Timed, public Output<Axon<ConTypePack>, float, ConTypePack,
                                                       typename PreRulePacker<typename ConTypePack::BaseProcessTypes, Axon<ConTypePack>>::type> {
    friend class Output<Axon, float, ConTypePack, typename PreRulePacker<typename ConTypePack::BaseProcessTypes, Axon>::type>;

public:
    using OutputBase = Output<Axon, float, ConTypePack, typename PreRulePacker<typename ConTypePack::BaseProcessTypes, Axon>::type>;

    explicit Axon(Host* host) : Timed(host), OutputBase(host) {}

    Axon(Axon&&) noexcept = default;
    Axon& operator=(Axon&&) noexcept = default;

    explicit Axon(const Axon&) = delete;
    Axon& operator=(const Axon&) = delete;

    Network* network() const { return dynamic_cast<Network*>(this->host()); }

    // Construction Methods
    void setExcitatory(const bool isExcitatory = true) {
        isExcitatory_ = isExcitatory;

        if (isExcitatory_) { transmitter_ = Glutamate; }
        else { transmitter_ = GABA; }

        this->forEachOutput([&](auto& syn) {
            syn.setExcitatory(isExcitatory_);
        });
    }
    void setInhibitory(const bool isInhibitory = true) { setExcitatory(!isInhibitory); }
    bool isExcitatory() const { return isExcitatory_; }
    bool isInhibitory() const { return !isExcitatory_; }
    float delay() const { return delay_; }
    void setDelay(const  float delay) { delay_ = delay; }

    static std::string toString(const Transmitter transmitter) {
        switch (transmitter) {
            case Glutamate: return "Glutamate";
            case GABA: return "GABA";
            case NoTransmitter: return "None";
            default: return "Unknown";
        }
    }
    static Transmitter toTransmitter(const uint8_t transmitter) {
        if (transmitter == 0)  { return Glutamate; }
        if (transmitter == 1)  { return GABA; }
        if (transmitter == UINT8_MAX)  { return NoTransmitter; }
        return UnknownTransmitter;
    }
    void setTransmitter(const Transmitter transmitter) {
        transmitter_ = transmitter;
        if (transmitter_ == Glutamate) {
            setExcitatory();
        } else if (transmitter_ == GABA) {
            setInhibitory();
        }
    }
    std::string transmitterName() const { return toString(transmitter_); }

protected:
    /// Initiates the propagation of a spike. If delay > timestep the spike will be generated later
    void forward() {
        this->delayedUpdate([this] { transmit(); }, delay_);
    }

private:
    /// Send spike to output synapses and transmit. Causes internal and external change.
    void transmit() {
        this->forEachOutput([&](auto& syn) {
            syn.learn();
            syn.propagate();
        });
        lastTransmit_ = this->T();
    }
    /// Send spike to output synapses but do not transmit. Causes internal change only.
    void absorb() {
        if (this->T() > lastTransmit_) {   // this ensures that syn.fire is not called twice per timestep
            this->forEachOutput([&](auto& syn) {
                syn.learn();
            });
        }
    }

    bool isSpiking_ = false;
    bool isExcitatory_ = true;
    uint64_t lastTransmit_ = 0;
    float delay_ = 0.;

    Transmitter transmitter_ = Glutamate;
    Transmitter modTransmitter_ = NoTransmitter;
};


// Type traits
template <typename ConTypePack>
struct SignalType_t<Axon<ConTypePack>> {
    using type = SignalType_t<Output<Axon<ConTypePack>, float, ConTypePack,
                                     typename ProcessPackRepacker<ConTypePack>::type>>::type;
};


#endif //AXON_H
