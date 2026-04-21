/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#pragma once
#include <string>
#include <array>
#include <stdexcept>
#include <cstdint>
#include <sstream>
#include <iomanip>

static std::pair<uint64_t, uint64_t> uuidToUint64Pair(const std::string& uuid) {
    // Remove hyphens
    std::string hex;
    hex.reserve(32);
    for (char c : uuid) {
        if (c != '-') hex += c;
    }
    if (hex.size() != 32)
        throw std::invalid_argument("Invalid UUID string: " + uuid);

    uint64_t high = std::stoull(hex.substr(0, 16), nullptr, 16);
    uint64_t low  = std::stoull(hex.substr(16, 16), nullptr, 16);
    return {high, low};
}

static std::string uint64PairToUuid(uint64_t high, uint64_t low) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');

    // high: 8-4-4 groups (32 + 16 + 16)
    oss << std::setw(8) << ((high >> 32) & 0xFFFFFFFF) << '-';
    oss << std::setw(4) << ((high >> 16) & 0xFFFF)     << '-';
    oss << std::setw(4) << ( high        & 0xFFFF)     << '-';

    // low: 4-12 groups (16 + 48)
    oss << std::setw(4) << ((low >> 48) & 0xFFFF)      << '-';
    oss << std::setw(12) << (low        & 0xFFFFFFFFFFFFLL);
    return oss.str();
}

static std::string uint64PairToUuid(const std::pair<uint64_t, uint64_t>& p) {
    return uint64PairToUuid(p.first, p.second);
}