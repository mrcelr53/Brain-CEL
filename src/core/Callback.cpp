/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#include "Callback.h"


CallbackQueue::CallbackQueue(float *timestep, uint64_t *tick)
    : dt_(timestep), tick_(tick) {}
void CallbackQueue::schedule(const std::function<void()>& fn, const int delay = 1) {
    queue.push(Callback{*tick_ + delay, fn});
}
void CallbackQueue::tick() {
    while (!queue.empty() && queue.top().tick <= T()) {
        auto [time, action] = queue.top();
        queue.pop();
        action();
    }
}