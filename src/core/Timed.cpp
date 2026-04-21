/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#include "core/Timed.h"

#include "Host.h"

int Timed::id() const {
    return id_ + idOffset_;
};
int Timed::localId() const {
    return id_;
}
void Timed::setLocalId(const int localId) {
    id_ = localId;
    onIdSet(id(), idOffset());
}
void Timed::setLocalId(const int localId, const int offset) {
    setIdOffset(offset);
    setLocalId(localId);
    onIdSet(id(), idOffset());
}
void Timed::setId(const int globalId) {
    setLocalId(globalId - idOffset_);
    onIdSet(id(), idOffset());
}
void Timed::setId(const int globalId, const int offset) {
    setIdOffset(offset);
    setId(globalId);
    onIdSet(id(), idOffset());
}
void Timed::setIdOffset(const int offset) {
    idOffset_ = offset;
    onIdSet(id(), idOffset());
}

Timed::Timed(Host *host) : host_(host) {
    Timed::synchronize(host->context());
}
void Timed::synchronize(SimulationContext *ctx) {
    if (ctx) {
        dt_ = ctx->timestep;
        tick_ = ctx->tick;
        updateQueue_ = ctx->updateQueue_;
        pushQueue_ = ctx->pushQueue_;
    }
}