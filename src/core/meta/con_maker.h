/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#ifndef CON_MAKER_H
#define CON_MAKER_H

#include "type_pack.h"
#include "type_list.h"
#include "con_expander.h"



/**
 * Its non-trivial to instantiate a templated connection.
 * Every connection requires the full TypeVectorPack of possible pre- and post-sites.
 * This creates a circular dependency, since any Output- or Input-Module also require a full TypeVectorPack
 * of all possible Connection types.
 *
 * The ConnectionMaker resolves the circular dependencies with forward declaration.
 * The connections are expanded following the cartesian product.
 *
 * Example:
 * - Pre Terminals  [ A, B ]
 * - Post Terminals [ D, E, F ]
 * - Resulting Connections: [ A->D, A->E, A->F, B->D, B->E, B->F ]
 */
template <
    template <typename, typename, typename, bool> class ConKind,
    typename PreTerminals,
    typename PostTerminals,
    typename ProcessTypes = TypeList<>,
    bool LearningPossible = false,
    template <typename...> class PackKind = TypeVectorPack
>
struct ConnectionMaker {
    template <typename Self>
    struct Maker {
        // Get expanded lists from builders
        using PreTypes = PreTerminals::template apply<Self>;
        using PostTypes = PostTerminals::template apply<Self>;

        // Expand to all combinations
        using ExpandedConnections = Expander<ConKind, PreTypes, PostTypes, ProcessTypes, LearningPossible>::type;

        struct type : ListToPack<PackKind, ExpandedConnections>::type {
            using BaseProcessTypes = ProcessTypes;

            // Forward storage types
            template <bool IsPointerStorage = false>
            using Storage = ListToPack<PackKind, ExpandedConnections>::template Storage<IsPointerStorage>;
        };
    };
};




#endif //CON_MAKER_H
