/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef STORAGE_H
#define STORAGE_H

#include <bitset>
#include <unordered_set>
#include <vector>

#include <braincel/Log.h>

#include "meta/type_pack.h"


// Concept for forEach
template <typename F, typename Pack>
concept VoidCallableForPack =
    requires(F f, Pack* p) {
        { []<typename... Ts>(TypePack<Ts...>*) -> bool {
            return (std::invocable<F, Ts&> && ...) && (std::same_as<std::invoke_result_t<F, Ts&>, void> && ...);
        } (p) } -> std::convertible_to<bool>;
    } ||
    requires(F f, Pack* p) {
        { []<typename... Ts>(TypeVectorPack<Ts...>*) -> bool {
            return (std::invocable<F, Ts&> && ...) && (std::same_as<std::invoke_result_t<F, Ts&>, void> && ...);
        } (p) } -> std::convertible_to<bool>;
    } ||
    requires(F f, Pack* p) {
        { [] (Pack*) -> bool {
            using Tuple = typename Pack::ProcessTuple;
            constexpr size_t N = std::tuple_size_v<Tuple>;
            return ([&]<size_t... Is>(std::index_sequence<Is...>) constexpr {
                return ((std::invocable<F, std::tuple_element_t<Is, Tuple>&> &&
                         std::same_as<std::invoke_result_t<F, std::tuple_element_t<Is, Tuple>&>, void>) && ...);
            } (std::make_index_sequence<N>{}));
        } (p) } -> std::convertible_to<bool>;
    };

// Concept for forEach
template <typename F, typename Pack>
concept BoolCallableForPack =
    requires(F f, Pack* p) {
        { []<typename... Ts>(TypePack<Ts...>*) -> bool {
            return (std::invocable<F, Ts&> && ...) && (std::same_as<std::invoke_result_t<F, Ts&>, bool> && ...);
        } (p) } -> std::convertible_to<bool>;
    } ||
    requires(F f, Pack* p) {
        { []<typename... Ts>(TypeVectorPack<Ts...>*) -> bool {
            return (std::invocable<F, Ts&> && ...) && (std::same_as<std::invoke_result_t<F, Ts&>, bool> && ...);
        } (p) } -> std::convertible_to<bool>;
    } ||
    requires(F f, Pack* p) {
        { [] (Pack*) -> bool {
            using Tuple = typename Pack::ProcessTuple;
            constexpr size_t N = std::tuple_size_v<Tuple>;
            return ([&]<size_t... Is>(std::index_sequence<Is...>) constexpr {
                return ((std::invocable<F, std::tuple_element_t<Is, Tuple>&> &&
                         std::same_as<std::invoke_result_t<F, std::tuple_element_t<Is, Tuple>&>, bool>) && ...);
            } (std::make_index_sequence<N>{}));
        } (p) } -> std::convertible_to<bool>;
    };



struct StorageHandle {
    void* ptr;
    size_t type_idx;
};


/**
 * Handles one instance per type, with optional external insert or owned emplace.
 * Owned storage is type-safe per-type vectors of unique_ptr.
 */
template <typename TypePack, bool IsPointerStorage = true>
struct MixedStorage {
    using Storage = TypePack::template Storage<IsPointerStorage>;
    using OwnedStorage = TypePack::OwnedStorage;
    static constexpr bool PointerStorage = IsPointerStorage;
    static constexpr size_t NumTypes = std::tuple_size_v<Storage>;
    static constexpr size_t NumOwnedTypes = std::tuple_size_v<OwnedStorage>;

    MixedStorage() = default;

    MixedStorage(MixedStorage&&) noexcept = default;
    MixedStorage& operator=(MixedStorage&&) noexcept = default;

    MixedStorage(const MixedStorage&) = delete;
    MixedStorage& operator=(const MixedStorage&) = delete;

    template <typename Type>
    auto& getSlot() {
        constexpr size_t idx = index_of_v<Type, TypePack>;
        return std::get<idx>(storage_);
    }

    template <typename Type>
    auto& getSlot() const {
        constexpr size_t idx = index_of_v<Type, TypePack>;
        return std::get<idx>(storage_);
    }

    template <typename Type>
    bool has() const {
        constexpr size_t idx = index_of_v<Type, TypePack>;
        return contains_[idx];
    }

    template <typename Type>
    Type& get() {
        constexpr size_t idx = index_of_v<Type, TypePack>;
        if (!contains_[idx]) { BC_ERROR_ONCE("Storage", "process type not present in MixedStorage"); }

        auto& slot = getSlot<Type>();
        if constexpr (IsPointerStorage) { return **slot; }
        else { return slot.value(); }
    }

    template <typename Type>
    const Type& get() const {
        constexpr size_t idx = index_of_v<Type, TypePack>;
        if (!contains_[idx]) { BC_ERROR_ONCE("Storage", "process type not present in MixedStorage"); }

        auto& slot = getSlot<Type>();
        if constexpr (IsPointerStorage) { return **slot; }
        else { return slot.value(); }
    }

    template <typename Type>
    Type* getPtr() {
        constexpr size_t idx = index_of_v<Type, TypePack>;
        if (!contains_[idx]) { return nullptr; }

        auto& slot = getSlot<Type>();
        if constexpr (IsPointerStorage) { return *slot; }
        else { return &slot.value(); }
    }

    template <typename Type>
    void insert(Type& obj) {
        constexpr size_t idx = index_of_v<Type, TypePack>;
        if (!contains_[idx]) {
            contains_[idx] = true;
            count_++;
        }

        auto& slot = getSlot<Type>();
        if constexpr (IsPointerStorage) { slot = &obj; }
        else { slot = obj; }
    }

    template <typename Type, typename... Args>
    Type& emplace(Args&&... args) {
        constexpr size_t idx = index_of_v<Type, TypePack>;
        if (!contains_[idx]) {
            contains_[idx] = true;
            count_++;
        }

        if constexpr (IsPointerStorage) {
            auto& ownedPtr = std::get<std::unique_ptr<Type>>(ownedStorage_);
            ownedPtr = std::make_unique<Type>(std::forward<Args>(args)...);
            Type* raw = ownedPtr.get();
            getSlot<Type>() = raw;
            return *raw;
        }
        else {
            auto& slot = getSlot<Type>();
            slot.emplace(std::forward<Args>(args)...);
            return slot.value();
        }
    }

    template <typename Type>
    void remove() {
        constexpr size_t idx = index_of_v<Type, TypePack>;
        if (contains_[idx]) {  // Idempotent: decrement only if present
            contains_[idx] = false;
            count_--;

            auto& slot = getSlot<Type>();
            if constexpr (IsPointerStorage) {
                Type* ptr = *slot;
                auto& ownedPtr = std::get<std::unique_ptr<Type>>(ownedStorage_);
                if (ownedPtr && ownedPtr.get() == ptr) {
                    ownedPtr.reset();
                }
            }
            slot.reset();
        }
    }

    template <typename F>
        requires (VoidCallableForPack<F, TypePack> || BoolCallableForPack<F, TypePack>)
    void forEach(F&& func) {
        if constexpr (VoidCallableForPack<F, TypePack>) {
            forEachVoidHelper(std::make_index_sequence<NumTypes>{}, std::forward<F>(func));
        }
        else {
            forEachBoolHelper(std::make_index_sequence<NumTypes>{}, std::forward<F>(func));
        }
    }
    template <typename F>
        requires (VoidCallableForPack<F, TypePack> || BoolCallableForPack<F, TypePack>)
    void forEach(F&& func) const {
        if constexpr (VoidCallableForPack<F, TypePack>) {
            forEachVoidHelper(std::make_index_sequence<NumTypes>{}, std::forward<F>(func));
        } else {
            (void) forEachBoolHelper(std::make_index_sequence<NumTypes>{}, std::forward<F>(func));
        }
    }

    void clear() {
        contains_.reset();
        count_ = 0;

        if constexpr (NumTypes > 0) {
            clearImpl(std::make_index_sequence<NumTypes>{});
        }
        if constexpr (NumOwnedTypes > 0) {
            clearImpl(std::make_index_sequence<NumOwnedTypes>{});
        }
    }

    template <typename Type>
    bool contains(const Type& obj) const {
        constexpr size_t idx = index_of_v<Type, TypePack>;
        if (!contains_[idx]) return false;

        const auto& slot = getSlot<Type>();
        if constexpr (IsPointerStorage) { return *slot == &obj; }
        else { return slot.value() == obj; }
    }

    bool empty() const {
        return count_ == 0; //contains_.none();
    }

    size_t size() const {
        return count_; //contains_.count();
    }

private:
    Storage storage_{};
    OwnedStorage ownedStorage_{};
    std::bitset<NumTypes> contains_{};
    uint8_t count_ = 0;

    template <typename F, size_t... Is>
    void forEachVoidHelper(std::index_sequence<Is...>, F&& func) {
        if constexpr (sizeof...(Is) == 0) {
            return;
        } else {
            ([&] {
                if (!contains_.test(Is)) return 0;
                auto& slot = std::get<Is>(storage_);
                if constexpr (PointerStorage) {
                    std::invoke(std::forward<F>(func), **slot);
                } else {
                    std::invoke(std::forward<F>(func), slot.value());
                }
                return 0;
            }() + ...);
        }
    }

    template <typename F, size_t... Is>
    void forEachVoidHelper(std::index_sequence<Is...>, F&& func) const {
        if constexpr (sizeof...(Is) == 0) {
            return;
        } else {
            ([&] {
                if (!contains_.test(Is)) return 0;
                const auto& slot = std::get<Is>(storage_);
                if constexpr (PointerStorage) {
                    std::invoke(std::forward<F>(func), **slot);
                } else {
                    std::invoke(std::forward<F>(func), slot.value());
                }
                return 0;
            }() + ...);
        }
    }

    template <typename F, size_t... Is>
    bool forEachBoolHelper(std::index_sequence<Is...>, F&& func) {
        if constexpr (sizeof...(Is) == 0) { return false; }
        return ( [&] {
            if (!contains_.test(Is)) return true;
            auto& slot = std::get<Is>(storage_);
            if constexpr (PointerStorage) {
                return std::invoke(std::forward<F>(func), **slot);
            }
            else {
                return std::invoke(std::forward<F>(func), slot.value());
            }
        }() && ... );
    }
    template <typename F, size_t... Is>
    bool forEachBoolHelper(std::index_sequence<Is...>, F&& func) const {
        if constexpr (sizeof...(Is) == 0) { return false; }
        return ( [&] {
            if (!contains_.test(Is)) return true;
            const auto& slot = std::get<Is>(storage_);
            if constexpr (PointerStorage) {
                return std::invoke(std::forward<F>(func), **slot);
            }
            else {
                return std::invoke(std::forward<F>(func), slot.value());
            }
        }() && ... );
    }

    template <size_t... Is>
    void clearHelper(std::index_sequence<Is...>) {
        (std::get<Is>(storage_).reset(), ...);
        (std::get<Is>(ownedStorage_).reset(), ...);
    }
};


/**
 * Handles multiple instances per type, either owning (values) or non-owning (pointers).
 * Reuses fold expressions for totals and iteration.
 * IsPointerStorage = false
 * @tparam TypeVectorPack:   Contains all possible types that the MixedMultiStorage can store e.g.
 *                           TypeVectorPack<TypeA, TypeB, ...>.
 * @tparam IsPointerStorage: false (default) - better for locality and loops across the storage.
 *                           Deactivates StorageHandle operations e.g. forEachHandle to avoid dangling pointers.
 *                           true - better for frequent mutations or large Types or StorageHandle operations.
 * @tparam HashesLookups:    false (default) - no hashing.
 *                           true - faster contains and modification with added memory cost.
 */
template <typename TypeVectorPack, bool IsPointerStorage = false, bool HashesLookups = false>
struct MixedMultiStorage {
    using Storage = TypeVectorPack::template Storage<IsPointerStorage>;
    static constexpr size_t NumTypes = std::tuple_size_v<Storage>;

    MixedMultiStorage() = default;
    MixedMultiStorage(MixedMultiStorage&&) noexcept = default;
    MixedMultiStorage& operator=(MixedMultiStorage&&) noexcept = default;
    MixedMultiStorage(const MixedMultiStorage&) = delete;
    MixedMultiStorage& operator=(const MixedMultiStorage&) = delete;

    // Aliases
    static constexpr bool PointerStorage = IsPointerStorage;
    using SupportsHandles = std::bool_constant<IsPointerStorage>;

    template <typename Type>
    auto& get() {
        if constexpr (IsPointerStorage) {
            return std::get<std::vector<Type*>>(storage_);
        } else {
            return std::get<std::vector<Type>>(storage_);
        }
    }

    template <typename Type>
    const auto& get() const {
        if constexpr (IsPointerStorage) {
            return std::get<std::vector<Type*>>(storage_);
        } else {
            return std::get<std::vector<Type>>(storage_);
        }
    }

    Timed* getBase(const size_t id) {
        return idMap_.get_id(id);
    }
    const Timed* getBase(const size_t id) const {
        return idMap_.get_id(id);
    }

    template <typename Type>
    auto& at(size_t localIndex) {
        auto& vec = get<Type>();
        if (localIndex >= vec.size()) {
            BC_DEBUG("Storage", "MixedMultiStorage::at({}) out of bounds (size {})", localIndex, vec.size());
        }
        if constexpr (IsPointerStorage) {
            return *vec[localIndex];
        } else {
            return vec[localIndex];
        }
    }

    template <typename Type>
    const auto& at(size_t localIndex) const {
        auto& vec = get<Type>();
        if (localIndex >= vec.size()) {
            BC_DEBUG("Storage", "MixedMultiStorage::at({}) out of bounds (size {})", localIndex, vec.size());
        }
        if constexpr (IsPointerStorage) {
            return *vec[localIndex];
        } else {
            return vec[localIndex];
        }
    }

    template <typename Type>
    size_t size() const {
        return get<Type>().size();
    }

    size_t size() const {
        return total_size_;
    }

    template <typename Type>
    size_t capacity() const {
        return get<Type>().capacity();
    }

    size_t capacity() const {
        size_t count = 0;
        std::apply([&](const auto&... vecs) {
            ((count += vecs.capacity()), ...);
        }, storage_);
        return count;
    }

    template <typename Type>
    void reserve(const size_t num) {
        get<Type>().reserve(num);
    }

    template <typename Type>
    void insert(Type& obj) {
        auto& vec = get<Type>();
        if constexpr (IsPointerStorage) {
            vec.push_back(&obj);
        }
        else {
            static_assert(std::is_copy_constructible_v<Type>,
                          "Cannot insert non-copyable type when storing by value (IsPointerStorage = false)");
            vec.push_back(obj);
        }
        total_size_++;

        if constexpr (HashesLookups) {
            hashed_.template insertHash<Type>(obj);
        }

        // Insert base object Timed
        Timed* baseObj;
        if constexpr (IsPointerStorage) { baseObj = vec.back(); }
        else { baseObj = static_cast<Timed*>(&vec.back()); }
        idMap_.insert_id(baseObj->id(), baseObj);
    }

    template <typename Type>
    void insert(Type* ptr) {
        static_assert(IsPointerStorage, "Pointer insert not allowed for value storage (IsPointerStorage = false)");
        auto& vec = get<Type>();
        if (ptr) {
            vec.push_back(ptr);
            total_size_++;
        }
        if constexpr (HashesLookups) { hashed_.template insertHash<Type>(*ptr); }

        if (ptr) {
            // Insert base object Timed
            Timed* baseObj;
            if constexpr (IsPointerStorage) { baseObj = vec.back(); }
            else { baseObj = static_cast<Timed*>(&vec.back()); }
            idMap_.insert_id(baseObj->id(), baseObj);
        }
    }

    template <typename Type, typename... Args>
    Type& emplace(Args&&... args) {
        static_assert(!IsPointerStorage, "Cannot emplace into pointer storage (IsPointerStorage = true)");
        auto& vec = get<Type>();
        //assert(vec.size() < vec.capacity() && "Capacity reached - cannot emplace without reallocation");
        if (vec.size() >= vec.capacity()) {
            BC_ERROR("Storage", "capacity reached (size {}, capacity {}) - emplace would reallocate", vec.size(), vec.capacity());
        }

        vec.emplace_back(std::forward<Args>(args)...);
        total_size_++;
        if constexpr (HashesLookups) { hashed_.template insertHash<Type>(vec.back()); }

        // Insert base object Timed
        Timed* baseObj;
        if constexpr (IsPointerStorage) { baseObj = vec.back(); }
        else { baseObj = static_cast<Timed*>(&vec.back()); }
        idMap_.insert_id(baseObj->id(), baseObj);
        return vec.back();
    }

    bool contains(const size_t id) {
        return idMap_.contains_id(id);
    }

    bool contains(const int id) const {
        return idMap_.contains_id(static_cast<size_t>(id));
    }

    template <typename Type>
    bool contains(const Type& obj) const {
        if constexpr (HashesLookups) {
            return hashed_.template has<Type>(obj);
        } else {
            return containsIf<Type>([&obj](const Type& item) {
                return item == obj;
            });
        }
    }

    template <typename Type>
    bool remove(Type& obj) {
        if constexpr (HashesLookups) {
            if (!hashed_.template has<Type>(obj)) { return false; }
        }
        return removeIf<Type>([&obj](const Type& item) {
            return item == obj;
        });
    }

    /// Slow removal but order stays consistent
    template <typename Type>
    bool removeAt(size_t localIndex) {
        auto& vec = get<Type>();
        if (localIndex < vec.size()) {
            const size_t memberId = IsPointerStorage ? (*vec[localIndex]).id() : vec[localIndex].id();
            Type val_copy = IsPointerStorage ? *vec[localIndex] : vec[localIndex];  // Copy value for hash erase
            vec.erase(vec.begin() + localIndex);
            total_size_--;
            if constexpr (HashesLookups) {
                hashed_.template eraseOne<Type>(val_copy);
            }
            idMap_.erase_id(memberId);
            return true;
        }
        return false;
    }

    /// Faster removal if order does not matter
    template <typename Type>
    bool swapRemoveAt(size_t localIndex) {
        auto& vec = get<Type>();
        if (localIndex < vec.size()) {
            const size_t memberId = IsPointerStorage ? (*vec[localIndex]).id() : vec[localIndex].id();
            Type val_copy = IsPointerStorage ? *vec[localIndex] : vec[localIndex];  // Copy value for hash erase
            if (localIndex != vec.size() - 1) {
                std::swap(vec[localIndex], vec.back());
                // No need to update hash for swapped item
            }
            vec.pop_back();
            total_size_--;
            if constexpr (HashesLookups) {
                hashed_.template eraseOne<Type>(val_copy);
            }
            idMap_.erase_id(memberId);
            return true;
        }
        return false;
    }

    template <typename Type>
    bool empty() const { return get<Type>().empty(); }

    bool empty() const { return total_size_ == 0; }

    template <typename F>
    void forEach(F&& func) {
        std::apply([&](auto&... vecs) {
            ([&] {
                if constexpr (IsPointerStorage) {
                    for (size_t i = 0; i < vecs.size(); ++i) {
                        auto* ptr = vecs[i];
                        if (ptr) {
                            __builtin_prefetch(ptr, 0, 1);  // Prefetch L1 read
                            std::invoke(std::forward<F>(func), *ptr);
                        }
                    }
                } else {
                    for (auto& item : vecs) { std::invoke(std::forward<F>(func), item); }
                }
            }(), ...);
        }, storage_);
    }

    template <typename F>
    void forEach(F&& func) const {
        std::apply([&](auto&... vecs) {
            ([&] {
                if constexpr (IsPointerStorage) {
                    for (size_t i = 0; i < vecs.size(); ++i) {
                        auto* ptr = vecs[i];
                        if (ptr) {
                            __builtin_prefetch(ptr, 0, 1);  // Prefetch L1 read
                            std::invoke(std::forward<F>(func), *ptr);
                        }
                    }
                } else {
                    for (auto& item : vecs) { std::invoke(std::forward<F>(func), item); }
                }
            }(), ...);
        }, storage_);
    }


    template <typename F>
    bool forEachWhile(F&& func) {
        return std::apply([&](auto&... vecs) {
            auto processVec = [&](auto& vec) -> bool {
                if constexpr (IsPointerStorage) {
                    for (auto ptr : vec) {
                        if (ptr && !std::invoke(std::forward<F>(func), *ptr)) { return false; }
                    }
                } else {
                    for (auto& item : vec) {
                        if (!std::invoke(std::forward<F>(func), item)) { return false; }
                    }
                }
                return true;
            };
            return (processVec(vecs) && ...);
        }, storage_);
    }
    template <typename F>
    bool forEachWhile(F&& func) const {
        return std::apply([&](const auto&... vecs) {
            auto processVec = [&](const auto& vec) -> bool {
                if constexpr (IsPointerStorage) {
                    for (auto ptr : vec) {
                        if (ptr && !std::invoke(std::forward<F>(func), *ptr)) { return false; }
                    }
                } else {
                    for (const auto& item : vec) {
                        if (!std::invoke(std::forward<F>(func), item)) { return false; }
                    }
                }
                return true;
            };
            return (processVec(vecs) && ...);
        }, storage_);
    }

    template <typename F>
        requires IsPointerStorage
    void forEachPointer(F&& func) {
        std::apply([&](const auto&... vecs) {
            ([&] {
                for (size_t i = 0; i < vecs.size(); ++i) {
                    if (auto* ptr = vecs[i]) {
                        __builtin_prefetch(ptr, 0, 1);  // Prefetch L1 read
                        std::invoke(std::forward<F>(func), ptr);
                    }
                }
            }(), ...);
        }, storage_);
    }

    template <typename F>
        requires IsPointerStorage
    void forEachPointer(F&& func) const {
        std::apply([&](const auto&... vecs) {
            ([&] {
                for (size_t i = 0; i < vecs.size(); ++i) {
                    if (auto* ptr = vecs[i]) {
                        __builtin_prefetch(ptr, 0, 1);  // Prefetch L1 read
                        std::invoke(std::forward<F>(func), ptr);
                    }
                }
            }(), ...);
        }, storage_);
    }

    template <typename F>
    void forEachEnumerate(F&& func) {
        std::apply([&](auto&... vecs) {
            auto process = [&](auto& vec) {
                for (size_t i = 0; i < vec.size(); ++i) {
                    if constexpr (IsPointerStorage) {
                        if (auto* ptr = vec[i]; ptr) {
                            std::invoke(std::forward<F>(func), i, *ptr);
                        }
                    } else {
                        std::invoke(std::forward<F>(func), i, vec[i]);
                    }
                }
            };
            (process(vecs), ...);
        }, storage_);
    }

    template <typename F>
    void forEachEnumerate(F&& func) const {
        std::apply([&](const auto&... vecs) {
            auto process = [&](const auto& vec) {
                for (size_t i = 0; i < vec.size(); ++i) {
                    if constexpr (IsPointerStorage) {
                        if (auto* ptr = vec[i]; ptr) {
                            std::invoke(std::forward<F>(func), i, *ptr);
                        }
                    } else {
                        std::invoke(std::forward<F>(func), i, vec[i]);
                    }
                }
            };
            (process(vecs), ...);
        }, storage_);
    }

    template <typename F>
    void forEachHandle(F&& func) {
        static_assert(IsPointerStorage, "forEachHandle requires pointer storage (IsPointerStorage=true) for stable handles");
        std::apply([&](auto&... vecs) {
            auto collectVec = [&](auto& vec, size_t& current_tid) {
                for (size_t i = 0; i < vec.size(); ++i) {
                    if constexpr (IsPointerStorage) {
                        if (auto* ptr = vec[i]; ptr) {
                            StorageHandle handle = {static_cast<void*>(ptr), current_tid};
                            std::invoke(std::forward<F>(func), i, handle, *ptr);
                        }
                    } else {
                        StorageHandle handle = {static_cast<void*>(vec[i]), current_tid};
                        std::invoke(std::forward<F>(func), i, handle, vec[i]);
                    }
                }
                ++current_tid;
            };
            size_t tid = 0;
            (collectVec(vecs, std::ref(tid)), ...);
        }, storage_);
    }

    template <typename F>
    void forEachHandle(F&& func) const {
        static_assert(IsPointerStorage, "forEachHandle requires pointer storage (IsPointerStorage=true) for stable handles");
        std::apply([&](const auto&... vecs) {
            auto collectVec = [&](const auto& vec, size_t& current_tid) {
                for (size_t i = 0; i < vec.size(); ++i) {
                    if constexpr (IsPointerStorage) {
                        if (auto* ptr = vec[i]; ptr) {
                            StorageHandle handle = {static_cast<void*>(ptr), current_tid};
                            std::invoke(std::forward<F>(func), i, handle, *ptr);
                        }
                    } else {
                        StorageHandle handle = {static_cast<void*>(vec[i]), current_tid};
                        std::invoke(std::forward<F>(func), i, handle, vec[i]);
                    }
                }
                ++current_tid;
            };
            size_t tid = 0;
            (collectVec(vecs, std::ref(tid)), ...);
        }, storage_);
    }

    template <typename F>
    void visitHandle(const StorageHandle& handle, F&& func) {
        if (handle.type_idx >= NumTypes) return;
        dispatchHandle<0>(handle, std::forward<F>(func));
    }

    template <typename F>
    void visitHandle(const StorageHandle& handle, F&& func) const {
        if (handle.type_idx >= NumTypes) return;
        dispatchHandle<0>(handle, std::forward<F>(func));
    }

    template <typename Type>
    void clear() {
        auto& vec = get<Type>();

        if constexpr (HashesLookups) {
            hashed_.template getHash<Type>().clear();
        }

        if constexpr (IsPointerStorage) {
            for (auto* ptr : vec) {
                if (ptr) { idMap_.erase_id(ptr->id()); }
            }
        }
        else {
            for (const auto& item : vec) {
                idMap_.erase_id(item.id());
            }
        }

        total_size_ -= vec.size();
        vec.clear();
    }

    void clear() {
        std::apply([](auto&... vecs) {
            ((vecs.clear()), ...);
        }, storage_);
        if constexpr (HashesLookups) {
            std::apply([](auto&... hash) {
                ((hash.clear()), ...);
            }, hashed_.hashStorage_);
        }
        idMap_.clear();
        total_size_ = 0;
    }

    template <typename Type, typename Pred>
    bool removeIf(Pred&& pred) {
        static_assert(std::is_invocable_v<Pred&&, const Type&>,
                      "Predicate must be invocable with const Type&");
        if constexpr (HashesLookups) {
            static_assert(std::is_copy_constructible_v<Type>, "Type must be copy-constructible for hashed removals");
        }
        auto& vec = get<Type>();
        auto it = std::find_if(vec.begin(), vec.end(), [&](const auto& item) {
            if constexpr (IsPointerStorage) {
                return item && std::invoke(std::forward<Pred>(pred), *item);
            } else {
                return std::invoke(std::forward<Pred>(pred), item);
            }
        });
        if (it != vec.end()) {
            const size_t memberId = IsPointerStorage ? (**it).id() : (*it).id();
            const Type& val_ref = IsPointerStorage ? **it : *it;
            Type val_copy = val_ref;  // Copy directly even if no default constructor
            vec.erase(it);
            total_size_--;
            if constexpr (HashesLookups) { hashed_.template eraseOne<Type>(val_copy); }
            idMap_.erase_id(memberId);
            return true;
        }
        return false;
    }
    template <typename Type, typename Pred>
    bool containsIf(Pred&& pred) const {
        static_assert(std::is_invocable_v<Pred&&, const Type&>,
                      "Predicate must be invocable with const Type&");
        const auto& vec = get<Type>();
        return std::any_of(vec.begin(), vec.end(), [&](const auto& item) {
            if constexpr (IsPointerStorage) {
                return item && std::invoke(std::forward<Pred>(pred), *item);
            } else {
                return std::invoke(std::forward<Pred>(pred), item);
            }
        });
    }

private:
    Storage storage_{};
    std::array<size_t, NumTypes+1> offsets_{};
    size_t total_size_ = 0;

    struct IdHashing {
        std::unordered_map<size_t, Timed*> id_map_{};

        void insert_id(size_t id, Timed* ref) {
            id_map_.insert({id, ref});
        }

        void erase_id(const size_t id) {
            id_map_.erase(id);
        }

        Timed* get_id(const size_t id) {
            const auto it = id_map_.find(id);
            if (it != id_map_.end()) {
                return it->second;
            }
            return nullptr;
        }

        const Timed* get_id(const size_t id) const {
            auto const it = id_map_.find(id);
            if (it != id_map_.end()) {
                return it->second;
            }
            return nullptr;
        }

        bool contains_id(const size_t id) const {
            return id_map_.contains(id);
        }

        void clear() {
            id_map_.clear();
        }

        IdHashing() = default;
        IdHashing(IdHashing&&) noexcept = default;
        IdHashing& operator=(IdHashing&&) noexcept = default;
        IdHashing(const IdHashing&) = delete;
        IdHashing& operator=(const IdHashing&) = delete;
    };
    IdHashing idMap_{};

    struct Hashing {
        using HashStorage = typename TypeVectorPack::HashStorage;
        HashStorage hashStorage_{};

        template <typename Type>
        auto& getHash() {
            return std::get<std::unordered_multiset<Type>>(hashStorage_);
        }

        template <typename Type>
        const auto& getHash() const {
            return std::get<std::unordered_multiset<Type>>(hashStorage_);
        }

        template <typename Type>
        void insertHash(const Type& val) {
            getHash<Type>().insert(val);
        }

        template <typename Type>
        bool has(const Type& obj) const {
            return getHash<Type>().count(obj) > 0;
        }

        template <typename Type>
        void eraseOne(const Type& obj) {
            auto& mset = getHash<Type>();
            auto it = mset.find(obj);
            if (it != mset.end()) {
                mset.erase(it);  // Erase one instance
            }
        }

        Hashing() = default;
        Hashing(Hashing&&) noexcept = default;
        Hashing& operator=(Hashing&&) noexcept = default;
        Hashing(const Hashing&) = delete;
        Hashing& operator=(const Hashing&) = delete;
    };
    struct NoHashing {};
    using HashComponent = std::conditional_t<HashesLookups, Hashing, NoHashing>;
    HashComponent hashed_{};


    void recomputeOffsets() {
        offsets_[0] = 0;
        size_t current = 0;
        auto sizes = std::apply([](const auto&... vecs) {
            return std::array{vecs.size()...};
        }, storage_);
        for (size_t i = 0; i < NumTypes; ++i) {
            current += sizes[i];
            offsets_[i+1] = current;
        }
    }

    std::pair<size_t, size_t> findPosition(size_t global) const {
        auto it = std::upper_bound(offsets_.cbegin(), offsets_.cend(), global);
        if (it == offsets_.cbegin()) {
            BC_ERROR_ONCE("Storage", "findPosition({}): global index below the first offset", global);
        }
        --it;  // points to the largest <= global
        size_t type_index = std::distance(offsets_.cbegin(), it);
        size_t local_index = global - offsets_[type_index];
        return {type_index, local_index};
    }

    template <size_t I, typename F>
    void dispatchHandle(StorageHandle& handle, F&& func, size_t global = 0) {
        if (handle.type_idx == I) {
            using Type = typename TypeVectorPack::template type_at<I>;
            using PtrType = std::conditional_t<IsPointerStorage, Type*, Type>;
            auto typed_ptr = static_cast<PtrType>(handle.ptr);
            if (typed_ptr) {
                std::invoke(std::forward<F>(func), global, *typed_ptr);
            }
            return;
        }
        if constexpr (I+1 < NumTypes) {     // recursive dispatch
            dispatchHandle<I+1>(handle, std::forward<F>(func), global);
        }
    }

    template <size_t I, typename F>
    void dispatchHandle(const StorageHandle& handle, F&& func, size_t global = 0) {
        if (handle.type_idx == I) {
            using Type = typename TypeVectorPack::template type_at<I>;
            using PtrType = std::conditional_t<IsPointerStorage, Type*, Type>;
            auto typed_ptr = static_cast<PtrType>(handle.ptr);
            if (typed_ptr) {
                std::invoke(std::forward<F>(func), global, *typed_ptr);
            }
            return;
        }
        if constexpr (I+1 < NumTypes) {   // recursive dispatch
            dispatchHandle<I+1>(handle, std::forward<F>(func), global);
        }
    }

};




#endif //STORAGE_H
