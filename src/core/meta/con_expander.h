/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef CON_EXPANDER_H
#define CON_EXPANDER_H


#include "type_list.h"
#include "con_process.h"


/// Expand all post-terminals inside the outer expander
template <template <typename, typename, typename, bool> class ConKind,
          typename Pre, typename PostList, typename RulePack, bool LearningEnabled = false>
struct InnerExpander;

template <template <typename, typename, typename, bool> class ConKind,
          typename Pre, typename RulePack, bool LearningEnabled>
struct InnerExpander<ConKind, Pre, TypeList<>, RulePack, LearningEnabled> {
    using type = TypeList<>;
};

template <template <typename, typename, typename, bool> class ConKind,
          typename Pre, typename PostBase, typename... PostRest,
          typename RulePack>
struct InnerExpander<ConKind, Pre, TypeList<PostBase, PostRest...>, RulePack, false> {
    // ConnectionProcessTypePack is a placeholder for the self-referential ProcessPack
    // ProcessPack = TypePack<ConnectionRule<RuleType, PreType, PostType, ConType=SelfRef!>
    using ProcessPack = ConnectionProcessTypePack<Pre, PostBase, RulePack, false, ConKind>;

    using ThisPost = ConKind<Pre, PostBase, ProcessPack, false>;
    using RecursePosts = InnerExpander<ConKind, Pre, TypeList<PostRest...>, RulePack, false>::type;
    using type = Append<TypeList<ThisPost>, RecursePosts>::type;
};

template <template <typename, typename, typename, bool> class ConKind,
          typename Pre, typename PostBase, typename... PostRest,
          typename RulePack>
struct InnerExpander<ConKind, Pre, TypeList<PostBase, PostRest...>, RulePack, true> {
    // ConnectionProcessTypePack is a placeholder for the self-referential ProcessPack
    // ProcessPack = TypePack<ConnectionRule<RuleType, PreType, PostType, ConType=SelfRef!>

    using ProcessPackFalse = ConnectionProcessTypePack<Pre, PostBase, RulePack, false, ConKind>;
    using ThisPostFalse = ConKind<Pre, PostBase, ProcessPackFalse, false>;

    using ProcessPackTrue = ConnectionProcessTypePack<Pre, PostBase, RulePack, true, ConKind>;
    using ThisPostTrue = ConKind<Pre, PostBase, ProcessPackTrue, true>;

    using BothThisPosts = TypeList<ThisPostFalse, ThisPostTrue>;

    using RecursePosts = InnerExpander<ConKind, Pre, TypeList<PostRest...>, RulePack, true>::type;
    using type = Append<BothThisPosts, RecursePosts>::type;
};

/// Expand all pre- and post-terminals. Pre terminals are expanded in the outer loop.
template <template <typename, typename, typename, bool> class ConKind,
          typename PreList, typename PostList, typename RulePack, bool LearningEnabled = false>
struct Expander;

template <typename PostList, typename RulePack, template <typename, typename, typename, bool> class ConKind,
          bool LearningEnabled>
struct Expander<ConKind, TypeList<>, PostList, RulePack, LearningEnabled> {
    using type = TypeList<>;
};

template <template <typename, typename, typename, bool> class ConKind,
          typename PreBase, typename... PreRest,
          typename PostList,
          typename RulePack,
          bool LearningEnabled>
struct Expander<ConKind, TypeList<PreBase, PreRest...>, PostList, RulePack, LearningEnabled> {
    using ThisPre = InnerExpander<ConKind, PreBase, PostList, RulePack, LearningEnabled>::type;
    using RecursePres = Expander<ConKind, TypeList<PreRest...>, PostList, RulePack, LearningEnabled>::type;
    using type = Append<ThisPre, RecursePres>::type;
};


#endif //CON_EXPANDER_H
