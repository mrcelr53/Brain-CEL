/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef LIST_H
#define LIST_H


#include "type_pack.h"


/// Type list helper
template <typename... Ts> struct TypeList {};

/// Returns a TypeList of all applied types
template <template <typename> class... Terminals>
struct TerminalList {
    template <typename Param>
    using apply = TypeList<Terminals<Param>...>;
};

/// Append helper for recursive concatenation
template <typename List1, typename List2> struct Append;

template <typename... Ts1, typename... Ts2>
struct Append<TypeList<Ts1...>, TypeList<Ts2...>> { using type = TypeList<Ts1..., Ts2...>; };

/// Map helper for applying a functor over TypeList
template <template <typename> class Mapper, typename List> struct Map;

template <template <typename> class Mapper, typename... Ts>
struct Map<Mapper, TypeList<Ts...>> { using type = TypeList<typename Mapper<Ts>::type...>; };

/// Convert TypeList to std::tuple
template <typename List> struct ListToTuple;

template <typename... Ts>
struct ListToTuple<TypeList<Ts...>> { using type = std::tuple<Ts...>; };

/// Unpack TypeList<Ts...> into PackKind<Ts...>
template <template <typename...> class PackKind, typename TypeList> struct ListToPack;

template <template <typename...> class PackKind, typename... Ts>
struct ListToPack<PackKind, TypeList<Ts...>> {
    using type = PackKind<Ts...>;
    template <bool IsPointerStorage = false>
    using Storage = typename type::template Storage<IsPointerStorage>;
};

#endif //LIST_H
