/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef CON_PROCESS_H
#define CON_PROCESS_H


#include "type_list.h"
#include "rule_expander.h"

/**
 * Self-referential ProcessPack factory to break circular dependency.
 * Mirrors TypePack interface: provides type_at<Idx>, Storage<IsPointerStorage>.
 * ConnectionRule inside refer to ConKind<Pre, Post, SelfRefConnectionProcesses>.
 *
 * Example:
 * @verbatim
 * Synapse<
 *   Axon<SynapsePack>,                               // PreType
 *   Soma<SynapsePack>,                               // PostType
 *   ConnectionProcessTypePack<                       // Self-ref factory
 *     Axon<SynapsePack>,
 *     Soma<SynapsePack>,
 *     TypeList<SparseSTDP, SparseLPL>,               // RuleList
 *     Synapse                                        // ConKind
 *   >
 * >
 * @endverbatim
 */
template <typename Pre, typename Post, typename RuleList, bool LearningEnabled,
          template <typename, typename, typename, bool> class ConKind>
struct ConnectionProcessTypePack {
    using ConType = ConKind<Pre, Post, ConnectionProcessTypePack, LearningEnabled>;

    template <typename Rule>
    struct CR { using type = ConnectionRule<Rule, Pre, Post, ConType>; };

    using ProcessList = Map<CR, RuleList>::type;           // TypeList<CR<Rule_i>...>
    using ProcessTuple = ListToTuple<ProcessList>::type;   // std::tuple<CR<Rule1>, CR<Rule2>, ...>

    template <size_t Idx>
    using type_at = std::tuple_element_t<Idx, ProcessTuple>;

    template <bool IsPointerStorage = true>
    using Storage = TupleStorage<ProcessTuple>::template type<IsPointerStorage>;
    using OwnedStorage = TupleOwnedStorage<ProcessTuple>::type;

    // For compatibility with SignalType_t (add specialization if needed, e.g.):
    // template <> struct SignalType_t<SelfRefConnectionProcesses<...>> { using type = ...; }
    // Or add: using underlying_tuple = ProcessTuple; and update SignalType_t to use it if present.
};


template <typename Type, typename Pack>
inline constexpr bool PackContains_v = index_of_v<Type, Pack> < PackSize_v<Pack>;


#endif //CON_PROCESS_H