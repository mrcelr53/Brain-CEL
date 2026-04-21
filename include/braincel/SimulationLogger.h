/*
* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#pragma once
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <unordered_map>
#include <nlohmann/json.hpp>

class SimulationLogger {
public:
    SimulationLogger();

    void setTargetFolder(const std::string& target);
    void initializeBatchDirectory(const std::string& prefix);
    void initializeDirectory(const std::string& prefix = "");

    std::string sessionFolder() const { return m_sessionFolder; }
    void setEnabled(bool enabled)     { m_enabled = enabled; }
    bool isEnabled()      const       { return m_enabled; }
    bool isInitialized()  const       { return !m_sessionFolder.empty(); }

    void logDataBatch(int tick, float biotime,
                      const std::vector<std::vector<float>>& data,
                      const std::string& type);

    void logScatterBatch(int tick, float biotime,
                         const std::vector<std::tuple<float, int, int>>& data,
                         const std::string& type);

    void logActivityBatch(int tick, float biotime,
                          const std::vector<float>& activity);

    void logSpikeIdBatch(int tick, float biotime,
                         const std::vector<std::vector<int>>& spikeIds);

    void logConnectome(int tick, float biotime,
                       const std::vector<std::vector<float>>& connectome);

    void logDistribution(int tick, float biotime,
                         const std::map<std::string, std::vector<std::vector<double>>>& distr,
                         const std::string& name);

    void logAbsDistribution(int tick, float biotime,
                            const std::map<int, std::vector<int>>& distr,
                            float minBin, float maxBin, float stepBin,
                            const std::string& name);

    void logFloatStates(int tick, float biotime, float x,
                        const std::unordered_map<std::string, float>& states,
                        const std::string& category);

    void logSimInfo(int tick, float biotime, const nlohmann::json& simInfo);
    void logParams(const std::string& name, const nlohmann::json& params);

private:
    std::string sanitize(const std::string& name);
    std::string makeDir(const std::string& path);
    std::string timestamp();

    std::string m_targetFolder = "/tmp/braincel_logs";
    std::string m_batchFolder;
    std::string m_sessionFolder;
    bool m_enabled = true;
};