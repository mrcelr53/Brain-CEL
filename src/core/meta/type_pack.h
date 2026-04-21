/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef PACK_H
#define PACK_H


#include <type_traits>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>


/// Type package which can store a single instance per class
template <typename... Ts>
struct TypePack {
    template <size_t Idx>
    using type_at = std::tuple_element_t<Idx, std::tuple<Ts...>>;

    template <bool IsPointerStorage = true>
    using Storage = std::tuple<std::optional<std::conditional_t<IsPointerStorage, Ts*, Ts>>...>;
    using OwnedStorage = std::tuple<std::unique_ptr<Ts>...>;
};

/// Type package which can store multiple instances per class
template <typename... Ts>
struct TypeVectorPack {
    template <size_t Idx>
    using type_at = std::tuple_element_t<Idx, std::tuple<Ts...>>;

    template <bool IsPointerStorage = true>
    using Storage = std::tuple<std::vector<std::conditional_t<IsPointerStorage, Ts*, Ts>>...>;
    using HashStorage = std::tuple<std::unordered_multiset<Ts>...>;  // Key is always the value type Ts
};



/// Helper to create Storage from a tuple type (emulating TypePack/TypeVectorPack)
template <typename TupleType> struct TupleStorage;

template <typename... Ts>
struct TupleStorage<std::tuple<Ts...>> {
    template <bool IsPointerStorage = true>
    using type = std::tuple<std::optional<std::conditional_t<IsPointerStorage, Ts*, Ts>>...>;
};



template <typename TupleType, typename IndexSeq> struct owned_storage_from_tuple;

template <typename TupleType, size_t... Is>
struct owned_storage_from_tuple<TupleType, std::index_sequence<Is...>> {
    using type = std::tuple<std::unique_ptr<std::tuple_element_t<Is, TupleType>>...>;
};
template <typename TupleType>
struct TupleOwnedStorage {
    static constexpr size_t Size = std::tuple_size_v<TupleType>;
    using type = typename owned_storage_from_tuple<TupleType, std::make_index_sequence<Size>>::type;
};



template <typename T>
struct SignalType_t;


template <typename... Ts>
struct SignalType_t<TypePack<Ts...>> {
private:
    static constexpr bool is_empty = sizeof...(Ts) <= 0;
    static_assert(!is_empty, "TypePack can not be empty to fetch SignalType");

    using FirstType = std::tuple_element_t<0, std::tuple<Ts...>>;
    using FirstSignal = typename SignalType_t<FirstType>::type;

    static_assert((std::is_same_v<typename SignalType_t<Ts>::type, FirstSignal> && ...),
                  "All types in TypePack must have the same signal type");

public:
    using type = FirstSignal;
};


template <typename... Ts>
struct SignalType_t<TypeVectorPack<Ts...>> {
private:
    static constexpr bool is_empty = sizeof...(Ts) <= 0;
    static_assert(!is_empty, "TypeVectorPack can not be empty to fetch SignalType");

    using FirstType = std::tuple_element_t<0, std::tuple<Ts...>>;
    using FirstSignal = typename SignalType_t<FirstType>::type;

    static_assert((std::is_same_v<typename SignalType_t<Ts>::type, FirstSignal> && ...),
                  "All types in TypeVectorPack must have the same signal type");

public:
    using type = FirstSignal;
};



template <size_t I, typename Pack, typename = void>
struct pack_element {};
template <size_t I, typename Pack>
struct pack_element<I, Pack, std::void_t<typename Pack::template type_at<I>>> {
    using type = Pack::template type_at<I>;
};

/// For accessing elements inside the pack
template <size_t I, typename Pack>
using PackElement_t = pack_element<I, Pack>::type;
template <typename Pack>
inline constexpr size_t PackSize_v = std::tuple_size_v<typename Pack::template Storage<false>>;

/// Get index of type
template <typename T, typename Pack, size_t I = 0, size_t N = PackSize_v<Pack>>
struct index_of_impl : std::integral_constant<size_t, N> {};
template <typename T, typename Pack, size_t I>
requires (I < PackSize_v<Pack>)
struct index_of_impl<T, Pack, I> : std::conditional_t<
    std::is_same_v<T, PackElement_t<I, Pack>>,
    std::integral_constant<size_t, I>,
    index_of_impl<T, Pack, I + 1>
> {};
template <typename T, typename Pack>
struct index_of : index_of_impl<T, Pack> {};
template <typename T, typename Pack>
inline constexpr size_t index_of_v = index_of<T, Pack>::value;

/// Check if pack contains a type
template<typename T, typename Pack>
struct is_in_pack : std::false_type {};
template<typename T, typename Pack>
inline constexpr bool is_in_pack_v = is_in_pack<T, Pack>::value;


#endif //PACK_H
