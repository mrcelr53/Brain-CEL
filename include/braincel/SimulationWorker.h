/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#pragma once

#include "build.h"
#include "SimulationLogger.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <atomic>
#include <chrono>

class Network;

class SimulationWorker {
public:
    SimulationWorker();
    ~SimulationWorker();

    void build();
    void simulate();
    void pause();
    void reset();
    void clear();

    void setBuildParams(const nlohmann::json& params)      { buildParams = params; }
    void setSimulationParams(const nlohmann::json& params) { simParams = params; }
    void simulationChanged(const nlohmann::json& params);
    void loadSimulationParams();

    bool buildCompleted() const { return buildComplete_; }
    void setBuildCompleted(bool v) { buildComplete_ = v; }

    Network* network() const { return net; }

    std::map<int, std::map<int, float>> getConnections(const std::string& preGroup,
                                                       const std::string& postGroup);

private:
    bool localBuild(const nlohmann::json& params);
    void localSimulate();
    void localPause();
    void localReset();
    void localClear();

    Network* net = nullptr;

    SimulationLogger logger{};
    int loggingSkip = 0;

    nlohmann::json simParams;
    nlohmann::json buildParams;

    std::string currentHost = "Localhost";
    bool buildComplete_ = false;

    std::atomic<bool> pause_{ false };
    bool flagSimulationChange_ = false;
    double lastBuildTime = 0.;
    int  tick_    = 0;
    float biotime_ = 0.f;

    // Sim params
    double duration               = 1000.0;
    float  timestep               = 1.0f;
    double sleepTime              = 0.0;
    double warmupTime             = 0.0;
    float  globalLearningFactor   = 1.0f;
    bool   randomDebugInput       = false;
    bool   log              = true;
    int    instancesPerModule     = 50;
    double instanceAmount         = 0.1;
    int    updateIntervalTicks    = 100;
    int    updateIntervalSpikeIds = 100;
    int    dftBinSize             = 1024;
    std::string valueState        = "Spikes";
    bool   absActivity            = true;
    bool   batchSimulation        = false;
    int    totalTicks             = 0;
    std::string device            = "GPU Compute";

    bool        doLog     = true;
    int         logSkips  = 0;
    std::string logPrefix = "sim";
};