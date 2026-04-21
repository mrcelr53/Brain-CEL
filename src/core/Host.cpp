/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#include "core/Host.h"


Host::Host(const std::string& name) : name_(name) {
}

void Host::advance(const double elapsedRealtime, const int elapsedTicks) {
    tick_ += elapsedTicks;

    if (tick_ % tickDurationUpdateInterval_ == 0) {
        const float fac = 1 / static_cast<float>(tickDurationUpdateInterval_);
        speedup_ = fac / static_cast<float>(currentRealtime()+1e-8);
        tickDurationAcc_ = 0.;
    }
    realtime_ += elapsedRealtime;
    tickDurationAcc_ += realtime_;
}
