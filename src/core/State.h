/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef STATE_H
#define STATE_H

#include <type_traits>
#include <utility>
#include <nlohmann/json.hpp>


#include "Schema.h"


struct State {
    virtual ~State() = default;
};

/// Deduce a states schema/value tuple types
template <typename T> using SchemaOf = decltype(T::schema());
template <typename T> using ValuesOf = decltype(std::declval<const T&>().values());

/// Default empty state
struct DefaultState final : State {};

/// Example non-reflecting state
struct ExampleState final : State {
    float tau1 = 0.f;
    float tau2 = 0.f;

    using Schema = std::tuple<Field<float>, Field<float>>;
    using Values = std::tuple<float, float>;

    static Schema schema() {
        return {
            Field<float>("tau1"),
            Field<float>("tau2")
        };
    }

    Values values() const {
        return {tau1, tau2};
    }
};




/// Reflecting State class which makes defining states with a lot of members less verbose
template <typename Derived>
struct ReflectingState : State {
    static constexpr auto& get_descriptions() {
        return Derived::descriptions;
    }

    // schema() returns the tuple of Field<T>...
    static constexpr auto schema() {
        return make_schema(std::make_index_sequence<
            std::tuple_size_v<std::remove_reference_t<decltype(get_descriptions())>>
        >{});
    }

    // values() returns the tuple of values in schema order
    auto values() const {
        return make_values(std::make_index_sequence<
            std::tuple_size_v<std::remove_reference_t<decltype(get_descriptions())>>
        >{});
    }

private:
    // Build schema tuple from descriptions
    template <size_t... Is>
    static constexpr auto make_schema(std::index_sequence<Is...>) {
        return std::make_tuple(
            Field<typename Member_t<decltype(std::get<Is>(get_descriptions()).ptr)>::type>{
                static_cast<std::string>(std::get<Is>(get_descriptions()).name)
            }...
        );
    }

    // Build values tuple from descriptions
    template <size_t... Is>
    auto make_values(std::index_sequence<Is...>) const {
        const Derived* self = static_cast<const Derived*>(this);
        return std::make_tuple(
            (self->*(static_cast<const typename Member_t<decltype(std::get<Is>(get_descriptions()).ptr)>::type Derived::*> (
                std::get<Is>(get_descriptions()).ptr
            )))...
        );
    }
};


struct ExampleReflectingState final : ReflectingState<ExampleReflectingState> {
    float tau1 = 0.f;
    float tau2 = 0.f;

    using Self = ExampleReflectingState;
    static constexpr auto descriptions = std::make_tuple(
        Member{ "tau1", &Self::tau1 },
        Member{ "tau2", &Self::tau2 }
    );
    using Schema = decltype(schema());                     // optional
    using Values = decltype(std::declval<Self>().values());      // optional
};



#endif //STATE_H
