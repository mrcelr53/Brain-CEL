/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef CONVERT_H
#define CONVERT_H

#include <unordered_map>
#include <string>
#include <cctype>
#include <vector>
#include <sstream>
#include <fstream>
#include <braincel/Log.h>
#include <iostream>


inline std::string convertStringToVariable(const std::string& input) {
    if (input.empty()) return "";

    std::string result;
    // Reserve space to avoid reallocations
    result.reserve(input.size() * 2);

    // Map of operators to their word replacements
    const std::unordered_map<char, std::string> operatorMap = {
        {'+', "_plus_"},
        {'-', "_minus_"},
        {'*', "_star_"},
        {'/', "_slash_"},
        {'%', "_percent_"},
        {'&', "_and_"},
        {'|', "_or_"},
        {'!', "_bang_"},
        {'=', "_equals_"},
        {'<', "_less_"},
        {'>', "_greater_"},
        {'^', "_caret_"},
        {'~', "_tilde_"},
        {'?', "_query_"},
        {':', "_colon_"}
    };

    bool lastWasUnderscore = false;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        // Handle operators
        if (operatorMap.find(c) != operatorMap.end()) {
            result += operatorMap.at(c);
            lastWasUnderscore = true;
            continue;
        }

        // Handle spaces and invalid characters
        if (std::isspace(c) || (!std::isalnum(c) && c != '_')) {
            if (!lastWasUnderscore) {
                result += '_';
                lastWasUnderscore = true;
            }
            continue;
        }

        // Handle alphanumeric characters
        if (std::isalnum(c)) {
            result += c;
            lastWasUnderscore = false;
        } else if (c == '_') {
            if (!lastWasUnderscore) {
                result += c;
                lastWasUnderscore = true;
            }
        }
    }

    // Remove trailing underscore
    if (!result.empty() && result.back() == '_') {
        result.pop_back();
    }

    // Ensure the variable name doesn't start with a digit
    if (!result.empty() && std::isdigit(result[0])) {
        result.insert(result.begin(), '_');
    }

    // If the result is empty after processing, return a default valid name
    if (result.empty()) {
        return "var";
    }

    return result;
}

template <typename T>
std::string vectorToString(const std::vector<T>& vec) {
    std::ostringstream oss;
    oss << "[ ";
    for (const auto& elem : vec) {
        oss << elem << " ";
    }
    oss << "]";
    return oss.str();
}
template <typename T>
std::string nestedVectorToString(const std::vector<std::vector<T>>& vec) {
    std::ostringstream oss;
    oss << "[ ";
    for (const auto& elem : vec) {
        oss << vectorToString(elem) << " ";
    }
    oss << "]";
    return oss.str();
}

inline std::string padToSize(const std::string& str, const size_t numBits) {
    std::string bitString = str;
    bitString.resize(numBits / 8, '\0'); // Pad with null characters
    return bitString;
}
inline std::string extractFromString(const std::string& str, const size_t startPos, const size_t numBits) {
    return str.substr(startPos / 8, numBits / 8);
}

inline std::string loadStringFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        BC_ERROR("IO", "failed to open file: {}", filePath);
        return "";
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

inline std::string colorToAnsi(const std::string& color) {
    int r = 255, g = 255, b = 255;

    if (color.size() >= 7 && color[0] == '#') {
        auto h = [&](int i) { return std::stoi(color.substr(i, 2), nullptr, 16); };
        r = h(1); g = h(3); b = h(5);
    }
    else if (color.rfind("rgb(", 0) == 0) {
        std::sscanf(color.c_str(), "rgb(%d, %d, %d)", &r, &g, &b);
    }
    else return "\033[37m"; // fallback white

    return "\033[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
}

#endif //CONVERT_H