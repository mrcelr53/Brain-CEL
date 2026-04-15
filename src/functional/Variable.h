/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef VARIABLE_H
#define VARIABLE_H

#include <optional>
#include <string>

#include "core/Connection.h"
#include "core/convert.h"


class Network;


template <typename PreType, typename PostType, typename ProcessTypePack = TypePack<>, bool EnableLearning = false>
class Variable final : public Connection<Variable<PreType, PostType, ProcessTypePack, EnableLearning>, PreType, PostType, ProcessTypePack, EnableLearning> {
    friend class Connection<Variable, PreType, PostType, ProcessTypePack, EnableLearning>;

public:
    using ConnectionBase = Connection<Variable, PreType, PostType, ProcessTypePack, EnableLearning>;
    using PreSignalType = SignalType_t<PreType>::type;
    using PostSignalType = SignalType_t<PostType>::type;

    Variable(Host* host, PreType& pre, PostType& post)
        : Timed(host), Connection<Variable, PreType, PostType, ProcessTypePack, EnableLearning>(host, pre, post) {}

    // Getters & Setters
    std::string className() const override {
        if (index_ >= 0) {
            return "[" + this->pre().className() + "]-> (" + std::to_string(index_) + ") ->[" + this->post().className() + "]";
        }
        return "[" + this->pre().className() + "]-->[" + this->post().className() + "]";
    }
    std::string className(const std::optional<std::tuple<int,int,int>>& preColor,
    const std::optional<std::tuple<int,int,int>>& postColor) const {
        auto preName = "[" + this->pre().className() + "]";
        auto postName = "[" + this->post().className() + "]";

        auto colorize = [](const std::string& text,
                           std::optional<std::tuple<int,int,int>> rgb) {
            if (!rgb) return text;
            auto [r,g,b] = *rgb;
            std::ostringstream ss;
            ss << "\033[48;2;" << r << ";" << g << ";" << b << "m"   // background color
               << text
               << "\033[0m";  // reset
            return ss.str();
        };

        preName = colorize(preName, preColor);
        postName = colorize(postName, postColor);

        if (index_ >= 0) {
            return preName + "-> (" + std::to_string(index_) + ") ->" + postName;
        }
        return preName + "-->" + postName;
    }

    int index() const { return index_; }
    void setIndex(const int index) { index_ = index; }

protected:
    void forward(const PreSignalType& signal) {
        if constexpr (std::is_same_v<PreSignalType, PostSignalType>) {
            // Match of pre- & post-signal types -> pass directly
            this->post().inject(signal);
            return;
        }

        constexpr bool preIsIndexable = requires(PreSignalType v) { v[0]; };
        constexpr bool postIsIndexable = requires(PostSignalType v) { v[0]; };

        if constexpr (preIsIndexable && !postIsIndexable) {
            // Fan-out
            if (index_ >= 0) {
                if (index_ < signal.size()) {
                    this->post().inject(signal[index_]);
                }
            } else {
                std::cerr << "Warning: Index not set in fan-out for " << className() << std::endl;
                this->post().inject(signal[0]);
            }
        }
        else if constexpr (!preIsIndexable && postIsIndexable) {
            // Broadcast
            if (index_ >= 0) {
                this->post().inject(index_, signal);
            } else {
                std::cerr << "Warning: Index not set in broadcast for " << className() << std::endl;
                this->post().inject(0, signal);
            }
        }
        else {
            std::cout << "Parameter::forward(const PreSignalType&) failed in " << className() << std::endl;
        }
    }

private:
    std::string name_;
    std::string key_;

    int index_ = -1;
};


template <typename ConTypePack, typename SignalType>
class TypedInput final : public Input<TypedInput<ConTypePack, SignalType>, SignalType, ConTypePack,
                                      typename ProcessPackRepacker<ConTypePack>::type> {
    friend class Input<TypedInput, SignalType, ConTypePack, typename ProcessPackRepacker<ConTypePack>::type>;

public:
    using InputBase = Input<TypedInput, SignalType, ConTypePack,
                            typename ProcessPackRepacker<ConTypePack>::type>;

    explicit TypedInput(Host *host) : Timed(host), InputBase(host) {}

    void resize(std::size_t size) {
        if constexpr (requires { value_.resize(size); }) {
            value_.resize(size);
        }
    }
    void resize(std::size_t size, const auto& defaultValue = {}) {
        if constexpr (requires { value_.resize(size, defaultValue); }) {
            value_.resize(size, defaultValue);
        }
    }

    // Getters & Setters
    SignalType value() const { return value_; }
    SignalType withdraw() { const auto val = value_; reinitialize(); return val; }
    void setValue(const SignalType& value) { value_ = value; }

    std::string name() const { return name_; }
    void setName(const std::string& name) { name_ = name; key_ = convertStringToVariable(name); }
    std::string key() const { return key_; }

protected:
    void reinitialize() {
        // Reset internal value
        if constexpr (std::is_arithmetic_v<SignalType>) {
            value_ = SignalType{0};
        }
        else if constexpr (requires { value_.fill(typename SignalType::value_type{0}); }) {
            value_.fill(typename SignalType::value_type{0});
        }
        else if constexpr (requires { value_.assign(value_.size(), typename SignalType::value_type{0}); }) {
            value_.assign(value_.size(), typename SignalType::value_type{0});
        }
        else if constexpr (requires { value_.clear(); }) {
            value_.clear();
        }
        else {
            value_ = SignalType{};
        }
    }
    void accumulate() {}
    void accumulate(const SignalType& signal) { value_ = signal; }
    template <typename SignalSubType>
    void accumulate(int idx, const SignalSubType& signal)
            requires requires(SignalType v, int i, SignalSubType& s) { { v[i] = s }; } {
        value_[idx] = signal;
    }

private:
    std::string name_;
    std::string key_;
    SignalType value_;
};


template <typename ConTypePack, typename SignalType>
class TypedOutput final : public Output<TypedOutput<ConTypePack, SignalType>, SignalType, ConTypePack,
                                        typename ProcessPackRepacker<ConTypePack>::type> {
    friend class Output<TypedOutput, SignalType, ConTypePack,
                        typename ProcessPackRepacker<ConTypePack>::type>;

public:
    using OutputBase = Output<TypedOutput, SignalType, ConTypePack,
                              typename ProcessPackRepacker<ConTypePack>::type>;

    explicit TypedOutput(Host *host) : Timed(host), OutputBase(host) {}

    // Getters & Setters
    SignalType value() const { return value_; }
    void setValue(const SignalType& value) { value_ = value; }

    std::string name() const { return name_; }
    void setName(const std::string& name) { name_ = name; key_ = convertStringToVariable(name); }
    std::string key() const { return key_; }

protected:
    void forward() {
        this->forEachOutput([&]<typename ConType>(ConType& con) {
            // Value is a scalar
            if constexpr (std::is_same_v<typename ConType::PreSignalType, SignalType>) {    // This check is required as the storage iterates over all kinds of connections_
                con.propagate(value_);
            }
        });
    }

private:
    SignalType value_;

    std::string name_;
    std::string key_;
};


// Type traits
template <typename ConTypePack, typename SignalType>
struct SignalType_t<TypedInput<ConTypePack, SignalType>> {
    using type = typename SignalType_t<Input<TypedInput<ConTypePack, SignalType>, SignalType, ConTypePack,
                                             typename ProcessPackRepacker<ConTypePack>::type>>::type;
};


template <typename ConTypePack, typename SignalType>
struct SignalType_t<TypedOutput<ConTypePack, SignalType>> {
    using type = typename SignalType_t<Output<TypedOutput<ConTypePack, SignalType>, SignalType, ConTypePack,
                                              typename ProcessPackRepacker<ConTypePack>::type>>::type;
};

#endif //VARIABLE_H
