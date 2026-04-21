/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#pragma once

#include <nlohmann/json.hpp>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <stdexcept>

enum class DistributionType {
    None,
    Uniform,
    Normal,
    LogNormal,
    Triangular
};

struct DistributionParams {
    DistributionType type = DistributionType::None;

    double mean     = 0.0;
    double stddev   = 1.0;
    double min      = 0.0;
    double max      = 1.0;
    double logMean  = 0.0;
    double logStddev= 1.0;
    double mode     = 0.5;

    bool isValid() const {
        switch (type) {
            case DistributionType::None:       return true;
            case DistributionType::Uniform:    return min <= max;
            case DistributionType::Normal:     return stddev > 0;
            case DistributionType::LogNormal:  return logStddev > 0;
            case DistributionType::Triangular: return min <= mode && mode <= max;
            default:                           return false;
        }
    }

    double getLogNormalMode() const {
        return std::exp(logMean - logStddev * logStddev);
    }

    double sample(std::mt19937& rng) const {
        switch (type) {
            case DistributionType::None:
                return mean;
            case DistributionType::Uniform: {
                std::uniform_real_distribution<double> d(min, max);
                return d(rng);
            }
            case DistributionType::Normal: {
                std::normal_distribution<double> d(mean, stddev);
                return d(rng);
            }
            case DistributionType::LogNormal: {
                std::lognormal_distribution<double> d(logMean, logStddev);
                return d(rng);
            }
            case DistributionType::Triangular: {
                std::uniform_real_distribution<double> u(0.0, 1.0);
                double v  = u(rng);
                double fc = (mode - min) / (max - min);
                if (v < fc)
                    return min + std::sqrt(v * (max - min) * (mode - min));
                else
                    return max - std::sqrt((1.0 - v) * (max - min) * (max - mode));
            }
            default:
                return mean;
        }
    }

    // String helpers

    static std::string typeToString(DistributionType t) {
        switch (t) {
            case DistributionType::None:       return "Fixed";
            case DistributionType::Uniform:    return "Uniform";
            case DistributionType::Normal:     return "Normal";
            case DistributionType::LogNormal:  return "Log-Normal";
            case DistributionType::Triangular: return "Triangular";
            default:                           return "Unknown";
        }
    }

    static DistributionType stringToType(const std::string& s) {
        if (s == "Fixed")      return DistributionType::None;
        if (s == "Uniform")    return DistributionType::Uniform;
        if (s == "Normal")     return DistributionType::Normal;
        if (s == "Log-Normal") return DistributionType::LogNormal;
        if (s == "Triangular") return DistributionType::Triangular;
        return DistributionType::None;
    }

    // JSON serialisation

    nlohmann::json toJson() const {
        if (type == DistributionType::None)
            return mean;

        nlohmann::json obj;
        obj["dist"] = typeToString(type);
        switch (type) {
            case DistributionType::Uniform:
                obj["min"] = min;
                obj["max"] = max;
                break;
            case DistributionType::Normal:
                obj["mean"]   = mean;
                obj["stddev"] = stddev;
                break;
            case DistributionType::LogNormal:
                obj["mode"]       = getLogNormalMode();
                obj["log_stddev"] = logStddev;
                break;
            case DistributionType::Triangular:
                obj["min"]  = min;
                obj["mode"] = mode;
                obj["max"]  = max;
                break;
            default:
                return mean;
        }
        return obj;
    }

    static DistributionParams fromJson(const nlohmann::json& j) {
        DistributionParams p;

        if (j.is_number()) {
            p.type = DistributionType::None;
            p.mean = j.get<double>();
            return p;
        }

        if (j.is_object() && j.contains("dist")) {
            p.type = stringToType(j["dist"].get<std::string>());

            switch (p.type) {
                case DistributionType::Uniform:
                    p.min  = j.value("min", 0.0);
                    p.max  = j.value("max", 1.0);
                    p.mean = (p.min + p.max) / 2.0;
                    break;
                case DistributionType::Normal:
                    p.mean   = j.value("mean",   0.0);
                    p.stddev = j.value("stddev",  1.0);
                    break;
                case DistributionType::LogNormal: {
                    double modeVal = j.value("mode",       1.0);
                    p.logStddev    = j.value("log_stddev", 0.5);
                    p.logMean      = std::log(modeVal) + p.logStddev * p.logStddev;
                    p.mean         = std::exp(p.logMean + 0.5 * p.logStddev * p.logStddev);
                    break;
                }
                case DistributionType::Triangular:
                    p.min  = j.value("min",  0.0);
                    p.mode = j.value("mode", 0.5);
                    p.max  = j.value("max",  1.0);
                    p.mean = (p.min + p.mode + p.max) / 3.0;
                    break;
                default:
                    p.type = DistributionType::None;
                    p.mean = j.value("mean", 0.0);
                    break;
            }
            return p;
        }

        // Fallback
        p.type = DistributionType::None;
        p.mean = 0.0;
        return p;
    }

    static bool isDistribution(const nlohmann::json& j) {
        return j.is_object() && j.contains("dist");
    }
};


class DistributionSampler {
public:
    DistributionSampler() : m_rng(std::random_device{}()) {}

    void setSeed(unsigned int seed) { seed_ = seed; m_rng.seed(seed); }
    unsigned int seed() const { return seed_; }

    // Sample from a JSON value — handles both plain numbers and distribution objects
    double sample(const nlohmann::json& j, double defaultValue = 0.0) {
        if (j.is_null() || j.is_discarded())
            return defaultValue;
        if (j.is_number())
            return j.get<double>();
        return DistributionParams::fromJson(j).sample(m_rng);
    }

    // Sample from DistributionParams directly
    double sample(const DistributionParams& params) {
        return params.sample(m_rng);
    }

    // Sample n values
    std::vector<double> sampleN(const nlohmann::json& j, int n) {
        std::vector<double> out;
        out.reserve(n);
        const auto params = DistributionParams::fromJson(j);
        for (int i = 0; i < n; ++i)
            out.push_back(params.sample(m_rng));
        return out;
    }

private:
    std::mt19937 m_rng;
    unsigned int seed_ = 0;
};