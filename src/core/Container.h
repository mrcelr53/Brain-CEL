/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef CONTAINER_H
#define CONTAINER_H

#ifdef slots
#undef slots
#endif

#include "core/Host.h"
#include "Timed.h"


/**
 *  Non-templated abstract base for polymorphic storage of any ModuleContainer
 *  (both Collections and Groups). Inheriting virtually from Timed allows
 *  diamond resolution if needed. Common operations are pure virtual here;
 *  templated implementations will provide concrete logic via CRTP for code sharing.
 */
class Container : public virtual Timed {
public:
    explicit Container(Host* host) : Timed(host) {}
    ~Container() override = default;

    virtual void load() = 0;
    virtual void save() = 0;
    virtual void reset() = 0;
};

/**
 * Non-templated base interface for Group, allowing polymorphic storage
 * of Group instances. The Group owns all modules and is responsible for simulation.
 */
class OwnerContainer : public Container {
public:
    explicit OwnerContainer(Host* host) : Timed(host), Container(host) {}

    virtual void update() = 0;
    virtual void push() = 0;
    virtual void pull() = 0;
    virtual void pushAll() = 0;
    virtual void pullAll() = 0;

    virtual void setCuda(bool cuda) = 0;

    // std::vector<size_t> updateActivity() const { return updateActivity_; }
    // std::vector<size_t> pullActivity() const { return pullActivity_; }
    // std::vector<size_t> pushActivity() const { return pushActivity_; }

protected:
    // std::vector<size_t> updateActivity_;
    // std::vector<size_t> pushActivity_;
    // std::vector<size_t> pullActivity_;

};

#endif //CONTAINER_H