/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#include <braincel/SimulationLogger.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <algorithm>

namespace fs = std::filesystem;

SimulationLogger::SimulationLogger() {}

std::string SimulationLogger::timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

std::string SimulationLogger::makeDir(const std::string& path) {
    fs::create_directories(path);
    return path;
}

std::string SimulationLogger::sanitize(const std::string& name) {
    std::string s = name;
    for (char& c : s) {
        if (c == ' ' || c == '-' || c == '/') c = '_';
    }
    return s;
}

void SimulationLogger::setTargetFolder(const std::string& target) {
    m_targetFolder = target;
    makeDir(m_targetFolder);
}

void SimulationLogger::initializeBatchDirectory(const std::string& prefix) {
    if (!m_batchFolder.empty()) return;
    m_batchFolder = m_targetFolder + "/batch_" + sanitize(prefix) + "_" + timestamp();
    makeDir(m_batchFolder);
}

void SimulationLogger::initializeDirectory(const std::string& prefix) {
    std::string base = m_batchFolder.empty() ? m_targetFolder : m_batchFolder;
    m_sessionFolder = base + "/" + sanitize(prefix) + "_" + timestamp();
    makeDir(m_sessionFolder);
}

void SimulationLogger::logDataBatch(int tick, float biotime,
                                    const std::vector<std::vector<float>>& data,
                                    const std::string& type) {
    if (!m_enabled || data.empty()) return;
    std::string dir = makeDir(m_sessionFolder + "/Data_" + sanitize(type));
    std::ofstream f(dir + "/tick_" + std::to_string(tick) + ".csv");
    f << "# biotime: " << biotime << "\n# type: " << type << "\n";
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i) f << ',';
            f << row[i];
        }
        f << '\n';
    }
}

void SimulationLogger::logScatterBatch(int tick, float biotime,
                                       const std::vector<std::tuple<float, int, int>>& data,
                                       const std::string& type) {
    if (!m_enabled || data.empty()) return;
    std::string dir = makeDir(m_sessionFolder + "/Scatter_" + sanitize(type));
    std::ofstream f(dir + "/tick_" + std::to_string(tick) + ".csv");
    f << "# biotime: " << biotime << "\ntime,neuron_id,module_id\n";
    for (const auto& [time, nid, mid] : data)
        f << time << ',' << nid << ',' << mid << '\n';
}

void SimulationLogger::logActivityBatch(int tick, float biotime,
                                        const std::vector<float>& activity) {
    if (!m_enabled || activity.empty()) return;
    std::string dir = makeDir(m_sessionFolder + "/Activity");
    std::ofstream f(dir + "/tick_" + std::to_string(tick) + ".csv");
    f << "# biotime: " << biotime << "\nsample_idx,activity\n";
    for (size_t i = 0; i < activity.size(); ++i)
        f << i << ',' << activity[i] << '\n';
}

void SimulationLogger::logSpikeIdBatch(int tick, float biotime,
                                       const std::vector<std::vector<int>>& spikeIds) {
    if (!m_enabled || spikeIds.empty()) return;
    std::string dir = makeDir(m_sessionFolder + "/SpikeIds");
    std::ofstream f(dir + "/tick_" + std::to_string(tick) + ".csv");
    f << "# biotime: " << biotime << "\n";
    for (const auto& ids : spikeIds) {
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) f << ',';
            f << ids[i];
        }
        f << '\n';
    }
}

void SimulationLogger::logConnectome(int tick, float biotime,
                                     const std::vector<std::vector<float>>& connectome) {
    if (!m_enabled || connectome.empty()) return;
    std::string dir = makeDir(m_sessionFolder + "/Connectome");
    std::ofstream f(dir + "/tick_" + std::to_string(tick) + ".csv");
    f << "# biotime: " << biotime << "\n";
    for (const auto& row : connectome) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i) f << ',';
            f << row[i];
        }
        f << '\n';
    }
}

void SimulationLogger::logDistribution(int tick, float biotime,
                                       const std::map<std::string, std::vector<std::vector<double>>>& distr,
                                       const std::string& name) {
    if (!m_enabled || distr.empty()) return;
    std::string dir = makeDir(m_sessionFolder + "/Distribution_" + sanitize(name));
    std::ofstream f(dir + "/tick_" + std::to_string(tick) + ".csv");
    f << "# biotime: " << biotime << "\n# distribution: " << name << "\n";
    for (const auto& [key, rows] : distr) {
        f << '[' << key << "]\n";
        for (const auto& row : rows) {
            for (size_t i = 0; i < row.size(); ++i) {
                if (i) f << ',';
                f << row[i];
            }
            f << '\n';
        }
    }
}

void SimulationLogger::logAbsDistribution(int tick, float biotime,
                                          const std::map<int, std::vector<int>>& distr,
                                          float minBin, float maxBin, float stepBin,
                                          const std::string& name) {
    if (!m_enabled || distr.empty()) return;
    std::string dir = makeDir(m_sessionFolder + "/AbsDistribution_" + sanitize(name));
    std::ofstream f(dir + "/tick_" + std::to_string(tick) + ".csv");
    f << "# biotime: " << biotime << "\n# bin_min: " << minBin
      << "\n# bin_max: " << maxBin << "\n# bin_step: " << stepBin
      << "\nbin_index,counts...\n";
    for (const auto& [binIdx, counts] : distr) {
        f << binIdx;
        for (int c : counts) f << ',' << c;
        f << '\n';
    }
}

void SimulationLogger::logFloatStates(int tick, float biotime, float x,
                                      const std::unordered_map<std::string, float>& states,
                                      const std::string& category) {
    if (!m_enabled || states.empty()) return;
    std::string dir = makeDir(m_sessionFolder + "/FloatStates_" + sanitize(category));
    std::ofstream f(dir + "/tick_" + std::to_string(tick) + ".csv");
    f << "# biotime: " << biotime << "\n# x: " << x << "\nname,value\n";
    for (const auto& [k, v] : states)
        f << k << ',' << v << '\n';
}

void SimulationLogger::logSimInfo(int tick, float biotime, const nlohmann::json& simInfo) {
    if (!m_enabled) return;
    std::string dir = makeDir(m_sessionFolder + "/SimInfo");
    std::ofstream f(dir + "/tick_" + std::to_string(tick) + ".csv");
    f << "# biotime: " << biotime << "\n";
    for (auto it = simInfo.begin(); it != simInfo.end(); ++it)
        f << it.key() << ',' << it.value().dump() << '\n';
}

void SimulationLogger::logParams(const std::string& name, const nlohmann::json& params) {
    if (!m_enabled) return;
    std::ofstream f(m_sessionFolder + "/" + sanitize(name) + ".json");
    f << params.dump(4);
}