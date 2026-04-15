/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef COVER_H
#define COVER_H


#include "Container.h"
#include "Storage.h"


constexpr bool ViewIsPointerStorage = true;
constexpr bool ViewHashesLookups = false;

/**
 * Non-owning View with member-pointers.
 */
template <typename MemberTypePack>
class View : public Container,
             public MixedMultiStorage<MemberTypePack, ViewIsPointerStorage, ViewHashesLookups> {
public:
    using Storage = typename MemberTypePack::template Storage<true>;
    using StorageBase = MixedMultiStorage<MemberTypePack, ViewIsPointerStorage, ViewHashesLookups>;

    explicit View(Host* host) : Timed(host), Container(host) {}

    View(View&&) noexcept = default;
    View& operator=(View&&) noexcept = default;

    explicit View(const View&) = delete;
    View& operator=(const View&) = delete;


    // Interface methods
    void load()  override { this->forEachPointer([](auto* module) { module->load(); }); }
    void save()  override { this->forEachPointer([](auto* module) { module->save(); }); }
    void reset() override { this->forEachPointer([](auto* module) { module->reset(); }); }
};

#endif // COVER_H
