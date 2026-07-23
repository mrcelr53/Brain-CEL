/*
* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#pragma once


#include <string>
#include <string_view>

#include <nlohmann/json.hpp>


// Metrics channel
class Metrics {
public:
    static void configureFromEnv();

    // Append
    static bool open(const std::string& path, bool append = true);
    static void close();
    static void flush();

    static bool active();

    // {"ev":"<ev>", ...record}
    static void emit(std::string_view ev, const nlohmann::json& record);
    static void emit(std::string_view ev);
};
