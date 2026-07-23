/*
* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#include <braincel/Metrics.h>

#include <cstdlib>
#include <fstream>
#include <mutex>

namespace {

std::mutex    g_mutex;
std::ofstream g_file;
bool          g_active = false;

} // namespace

void Metrics::configureFromEnv() {
    if (const char* path = std::getenv("BRAINCEL_METRICS"); path && *path) {
        open(path);
    }
}

bool Metrics::open(const std::string& path, const bool append) {
    std::lock_guard lock(g_mutex);
    if (g_file.is_open()) g_file.close();
    g_file.open(path, append ? std::ios::app : std::ios::trunc);
    g_active = g_file.is_open();
    return g_active;
}

void Metrics::close() {
    std::lock_guard lock(g_mutex);
    if (g_file.is_open()) { g_file.flush(); g_file.close(); }
    g_active = false;
}

void Metrics::flush() {
    std::lock_guard lock(g_mutex);
    if (g_file.is_open()) g_file.flush();
}

bool Metrics::active() { return g_active; }

void Metrics::emit(const std::string_view ev, const nlohmann::json& record) {
    if (!g_active) return;

    nlohmann::ordered_json out;    // ordered_json -> "ev" leads
    out["ev"] = std::string(ev);
    if (record.is_object()) {
        for (const auto& [key, value] : record.items()) {
            if (key != "ev") out[key] = value;
        }
    }

    std::lock_guard lock(g_mutex);
    if (!g_file.is_open()) return;
    
    // Flushed per record
    g_file << out.dump() << '\n';
    g_file.flush();
}

void Metrics::emit(const std::string_view ev) { emit(ev, nlohmann::json::object()); }
