/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef SCHEMA_H
#define SCHEMA_H

#include <nlohmann/json.hpp>


using json = nlohmann::json;

/// Forward declaration for recursive Field
template<typename T> struct Field;


/// Field for basic types and nested schemas
template<typename T>
struct Field {
    using type = T;
    explicit Field(std::string n) : name(std::move(n)) {}
    std::string name;
};

/// Specialization for nested schemas
template<typename... Fields>
struct Field<std::tuple<Fields...>> {
    using type = std::tuple<Fields...>;
    Field(std::string n, std::tuple<Fields...> f) : name(std::move(n)), fields(std::move(f)) {}
    std::string name;
    std::tuple<Fields...> fields;
};

/// Stores class member information
template <typename T, typename Class>
struct Member {
    std::string_view name;
    T Class::* ptr;
    constexpr Member(const std::string_view n, T Class::* p) : name(n), ptr(p) {}
};

/// Get the type from a member pointer
template <typename P>
struct Member_t;
template <typename T, typename C>
struct Member_t<T C::*> {
    using type = T;
};



/// Flatten schema into a tuple
template<typename T>
struct FlattenSchema;
template<typename T>
struct FlattenSchema<Field<T>> {
    using type = std::tuple<T>;
};
template<typename... Fields>
struct FlattenSchema<Field<std::tuple<Fields...>>> {
    using type = std::tuple<typename FlattenSchema<Fields>::type...>;
};
template<typename... Fields>
struct FlattenSchema<std::tuple<Fields...>> {
    using type = decltype(std::tuple_cat(std::declval<typename FlattenSchema<Fields>::type>()...));
};

/// Helper to convert type to schema
template<typename T>
json typeToJson() {
    if constexpr (std::is_same_v<T, float>) {
        return "float";
    } else if constexpr (std::is_same_v<T, double>) {
        return "double";
    } else if constexpr (std::is_same_v<T, int>) {
        return "integer";
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "string";
    } else if constexpr (std::is_same_v<T, bool>) {
        return "boolean";
    } else if constexpr (std::is_same_v<T, std::tuple<>>) {
        return json::object();
    } else if constexpr (std::is_same_v<T, std::vector<typename T::value_type>>) {
        return "array<" + typeToJson<typename T::value_type>().template get<std::string>() + ">";
    } else {
        return "unknown";
    }
}

/// Convert Field to schema
template<typename T>
json FieldToJson(const Field<T>& field) {
    return typeToJson<T>();
}

template<typename... Fields>
json FieldToJson(const Field<std::tuple<Fields...>>& field) {
    json obj = json::object();
    std::apply([&](const auto&... fs) {
        ((obj[fs.name] = FieldToJson(fs)), ...);
    }, field.fields);
    return obj;
}

/// Convert schema to JSON schema
template<typename SchemaType, size_t... Is>
json schemaToJson(const SchemaType& schema, std::index_sequence<Is...>) {
    json result = json::object();
    ((result[std::get<Is>(schema).name] = FieldToJson(std::get<Is>(schema))), ...);
    return result;
}

template<typename SchemaType>
json schemaToJson(const SchemaType& schema) {
    return schemaToJson(schema, std::make_index_sequence<std::tuple_size_v<SchemaType>>{});
}

/// Apply values to schema for JSON output
template<typename T, typename ValueType>
void applyField(json& result, const Field<T>& field, ValueType&& value) {
    result[field.name] = std::forward<ValueType>(value);
}

template<typename... Fields, typename... Values>
void applyField(json& result, const Field<std::tuple<Fields...>>& field, const std::tuple<Values...>& values) {
    json obj = json::object();
    std::apply([&](const auto&... fs) {
        std::apply([&](const auto&... vs) {
            size_t i = 0;
            ((obj[fs.name] = vs, i++), ...);
        }, values);
    }, field.fields);
    result[field.name] = obj;
}

template<typename SchemaType, typename TupleType, size_t... Is>
json apply(const SchemaType& schema, const TupleType& values, std::index_sequence<Is...>) {
    json result = json::object();
    size_t i = 0;
    ((applyField(result, std::get<Is>(schema), std::get<Is>(values)), i++), ...);
    return result;
}

template<typename SchemaType, typename TupleType>
json applySchema(const SchemaType& schema, const TupleType& values) {
    return apply(schema, values, std::make_index_sequence<std::tuple_size_v<SchemaType>>{});
}

#endif //SCHEMA_H