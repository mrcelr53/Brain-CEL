/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef SYNAPSE_H
#define SYNAPSE_H

#include <optional>
#include <string>
#include <sstream>

#include "core/meta/con_process.h"
#include "../core/Connection.h"
#include "Plasticity.h"

#include "generate.h"


class Network;


template <typename PreType, typename PostType, typename RuleTypePack = TypePack<>, bool EnableLearning = true>
class Synapse final : public Connection<Synapse<PreType, PostType, RuleTypePack, EnableLearning>, PreType, PostType, RuleTypePack, EnableLearning> {
    friend class Connection<Synapse, PreType, PostType, RuleTypePack, EnableLearning>;

public:
    Synapse(Host* host, PreType& pre, PostType& post)
        : Timed(host), Connection<Synapse, PreType, PostType, RuleTypePack, EnableLearning>(host, pre, post) {
        this->enableLearning(true);
        if (pre.isInhibitory()) {
            setInhibitory(true);
        } else {
            setInhibitory(false);
        }
    }

    Synapse(Synapse&&) noexcept = default;
    Synapse& operator=(Synapse&&) noexcept = default;

    explicit Synapse(const Synapse&) = delete;
    Synapse& operator=(const Synapse&) = delete;

    Network* network() const { return dynamic_cast<Network*>(this->host()); }

    void setWeight(const float weight, const float weight_std) {
        if (weight_std == 0.) { weight_ = weight; }
        else {
            std::lognormal_distribution dist(weight, weight_std);
            weight_ = dist(getRandomGenerator());
        }
    }
    void updateWeight(const float deltaWeight) {
        weight_ += deltaWeight * learningRate_;
        if (clipWeight_ && weight_ < 0.) { weight_ = 0; }
    }

    // Getters & Setters
    void setExcitatory(const bool excitatory = true) { excitatory_ = excitatory; }
    void setInhibitory(const bool inhibitory = true) { excitatory_ = !inhibitory; }
    bool isInhibitory() const { return !excitatory_; }
    bool isExcitatory() const { return excitatory_; }
    float weight() const { return weight_; }
    const float *weightPtr() const { return &weight_; }
    void setWeight(const float weight) { weight_ = weight; }
    void setMinWeight(const float minWeight) { minWeight_ = minWeight; }
    void setMaxWeight(const float maxWeight) { maxWeight_ = maxWeight; }
    float delay() const { return delay_; }
    void setDelay(const float delay) { delay_ = delay; }

    float learningRate() const { return learningRate_; }
    bool isLearning() const { return learningRate_ != 0 && this->connectionRule() != nullptr; }
    void setLearningRate(const float lr) {
        if (lr != 0.) { this->enableLearning(true); }
        else { this->enableLearning(false); }
        learningRate_ = lr;
        //std::cout << this->className() << " lr = " << lr << " " << this->isLearningEnabled() << std::endl;
    }

    bool doesWeightClip() const { return clipWeight_; }
    void setWeightClip(const bool clip) { clipWeight_ = clip; }

    std::string className() const override {
        std::string preName = "?"; std::string postName = "?";

        preName = this->pre().className();
        postName = this->post().className();

        if (isInhibitory()) { return "[" + preName + "]-> (-) >-[" + postName + "]"; }
        return "[" + preName + "]-> (+) >-[" + postName + "]";
    }
    std::string className(const std::optional<std::tuple<int,int,int>>& preColor,
                          const std::optional<std::tuple<int,int,int>>& postColor) const {
        std::string preName = "?"; std::string postName = "?";

        preName = "[" + this->pre().className() + "]";
        postName = "[" + this->post().className() + "]";

        auto colorize = [](const std::string& text,
                           std::optional<std::tuple<int,int,int>> rgb) {
            if (!rgb) return text;
            auto [r,g,b] = *rgb;
            std::ostringstream ss;
            ss << "\033[48;2;" << r << ";" << g << ";" << b << "m" << text << "\033[0m";
            return ss.str();
        };
        preName = colorize(preName, preColor); postName = colorize(postName, postColor);

        if (isInhibitory()) { return preName + "-> (-) >-" + postName; }
        return preName + "-> (+) >-" + postName;
    }

    template <typename ProcessType>
    ProcessType& process() {
        if constexpr (std::is_same_v<ProcessType, CR_STDPf>) {
            return *stdpfRule_;
        } else if constexpr (std::is_same_v<ProcessType, CR_STDP>) {
            return *stdpRule_;
        } else {
            static_assert(false, "Unsupported ProcessType in Synapse::process");
        }
    }

    template <typename ProcessType>
    const ProcessType& process() const {
        if constexpr (std::is_same_v<ProcessType, CR_STDPf>) {
            return *stdpfRule_;
        } else if constexpr (std::is_same_v<ProcessType, CR_STDP>) {
            return *stdpRule_;
        } else {
            static_assert(false, "Unsupported ProcessType in Synapse::process");
        }
    }

    template <typename ProcessType>
    bool hasProcess() const {
        if constexpr (std::is_same_v<ProcessType, CR_STDPf>) {
            return stdpfRule_.has_value();
        } else if constexpr (std::is_same_v<ProcessType, CR_STDP>) {
            return stdpRule_.has_value();
        } else {
            return false;
        }
    }

    template <typename ProcessType>
    bool containsProcess(const ProcessType& process) const {
        if constexpr (std::is_same_v<ProcessType, CR_STDPf>) {
            return stdpfRule_.has_value() && std::addressof(stdpfRule_.value()) == std::addressof(process);
        } else if constexpr (std::is_same_v<ProcessType, CR_STDP>) {
            return stdpRule_.has_value() && std::addressof(stdpRule_.value()) == std::addressof(process);
        } else {
            return false;
        }
    }

    size_t numProcesses() const {
        return (stdpfRule_.has_value() ? 1 : 0)
               + (stdpRule_.has_value()  ? 1 : 0);
    }

    template <typename ProcessType, typename... Args>
    ProcessType& emplaceProcess(Args&&... args) {
        if constexpr (std::is_same_v<ProcessType, CR_STDPf>) {
            stdpfRule_.emplace(this->host(), std::forward<Args>(args)...);
            return stdpfRule_.value();
        } else if constexpr (std::is_same_v<ProcessType, CR_STDP>) {
            stdpRule_.emplace(this->host(), std::forward<Args>(args)...);
            return stdpRule_.value();
        } else {
            static_assert(false, "Unsupported ProcessType in Synapse::emplaceProcess");
        }
    }

    // Minimal overrides for other management (extend as needed)
    template <typename ProcessType>
    void insertProcess(ProcessType& process) {
        static_assert(false, "insertProcess not supported in Synapse (use emplaceProcess)");
    }

    template <typename ProcessType>
    void removeProcess() {
        if constexpr (std::is_same_v<ProcessType, CR_STDPf>) {
            stdpfRule_.reset();
        } if constexpr (std::is_same_v<ProcessType, CR_STDP>) {
            stdpRule_.reset();
        } else {
            static_assert(false, "Unsupported ProcessType in Synapse::removeProcess");
        }
    }

    void clearProcesses() {
        stdpfRule_.reset();
        stdpRule_.reset();
    }

    template <typename F>
    void forEachProcess(F&& func) {
        if (stdpfRule_.has_value()) {
            std::invoke(std::forward<F>(func), stdpfRule_.value());
        }
        if (stdpRule_.has_value()) {
            std::invoke(std::forward<F>(func), stdpRule_.value());
        }
    }

    // Override updateProcesses to use direct dispatch (bypasses forEachProcess -> MixedStorage)
    void updateProcesses() {
        if (stdpfRule_.has_value()) stdpfRule_->update();
        if (stdpRule_.has_value()) stdpRule_->update();
    }


protected:
    void initialize() {
        weight_ = 1.;
    }
    void forward() {
        if (weight_ <= 0) { return; }
        if (excitatory_) {
            this->post().inject(weight_);
        } else {
            this->post().inject(-weight_);
        }
    }
    __attribute__((always_inline)) void adapt() {
        if (stdpfRule_.has_value()) stdpfRule_->update();
        if (stdpRule_.has_value()) stdpRule_->update();
    }

private:
    using CR_STDPf = ConnectionRule<SparseSTDPf, PreType, PostType, Synapse>;
    using CR_STDP = ConnectionRule<SparseSTDP, PreType, PostType, Synapse>;
    std::optional<CR_STDPf> stdpfRule_;
    std::optional<CR_STDP> stdpRule_;

    bool excitatory_ = true;
    bool clipWeight_ = true;

    float weight_ = 1.;
    float minWeight_ = 0.;
    float maxWeight_ = 10.;
    float learningRate_ = 0.1;
    float delay_ = 0.;
};




#endif // SYNAPSE_H
