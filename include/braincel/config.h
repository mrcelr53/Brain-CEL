/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef BRAINCEL_CONFIG_H
#define BRAINCEL_CONFIG_H

#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>


static nlohmann::json loadJson(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    return nlohmann::json::parse(f);
}
static void setField(nlohmann::json& root, const std::string& dotPath, const nlohmann::json& value) {
    nlohmann::json* cur = &root;
    std::istringstream ss(dotPath);
    std::string part;
    std::vector<std::string> parts;
    while (std::getline(ss, part, '.')) parts.push_back(part);

    for (size_t i = 0; i + 1 < parts.size(); ++i) {
        const auto& p = parts[i];
        if (cur->is_array()) {
            cur = &(*cur)[std::stoi(p)];
        } else {
            cur = &(*cur)[p];
        }
    }
    const auto& last = parts.back();
    if (cur->is_array()) (*cur)[std::stoi(last)] = value;
    else                 (*cur)[last]             = value;
}

#endif //BRAINCEL_CONFIG_H