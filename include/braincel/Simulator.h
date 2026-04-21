/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#pragma once
#include <nlohmann/json.hpp>
#include <string>

class SimulationWorker;

class Simulator {
public:
    Simulator();
    ~Simulator();

    void build(const nlohmann::json& params);
    void simulate(const nlohmann::json& params);
    void pause();
    void stop();

    void requestConnections(const std::string& preGroup, const std::string& postGroup);

private:
    SimulationWorker* worker = nullptr;
    bool buildComplete = false;
};