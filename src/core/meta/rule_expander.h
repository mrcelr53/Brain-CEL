/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef RULE_EXPANDER_H
#define RULE_EXPANDER_H


#include "type_list.h"

// Forward declarations
template <typename, typename> class PreRule;
template <typename, typename> class PostRule;
template <typename, typename, typename, typename> class ConnectionRule;


// Pre rule expander
template <typename RuleList, typename TerminalType>
struct PreRuleExpander;

template <typename TerminalType>
struct PreRuleExpander<TypeList<>, TerminalType> {
    using type = TypeList<>;
};

template <typename RuleBase, typename... RuleRest, typename TerminalType>
struct PreRuleExpander<TypeList<RuleBase, RuleRest...>, TerminalType> {
    using ThisRule = PreRule<RuleBase, TerminalType>;
    using Recurse = typename PreRuleExpander<TypeList<RuleRest...>, TerminalType>::type;
    using type = typename Append<TypeList<ThisRule>, Recurse>::type;
};

template <typename RuleList, typename TerminalType>
struct PreRulePacker {
    using RuleTypes = typename PreRuleExpander<RuleList, TerminalType>::type;
    using type = typename ListToPack<TypePack, RuleTypes>::type;
};



// Post rule expander
template <typename RuleList, typename TerminalType>
struct PostRuleExpander;

template <typename TerminalType>
struct PostRuleExpander<TypeList<>, TerminalType> {
    using type = TypeList<>;
};

template <typename RuleBase, typename... RuleRest, typename TerminalType>
struct PostRuleExpander<TypeList<RuleBase, RuleRest...>, TerminalType> {
    using ThisRule = PostRule<RuleBase, TerminalType>;
    using Recurse = typename PostRuleExpander<TypeList<RuleRest...>, TerminalType>::type;
    using type = typename Append<TypeList<ThisRule>, Recurse>::type;
};

template <typename RuleList, typename TerminalType>
struct PostRulePacker {
    using RuleTypes = typename PostRuleExpander<RuleList, TerminalType>::type;
    using type = typename ListToPack<TypePack, RuleTypes>::type;
};


// Connection rule expander
template <template <typename, typename, typename> class ConKind, typename Pre, typename Post, typename RulePack>
struct ConnectionRuleExpander;

template <template <typename, typename, typename> class ConKind, typename Pre, typename Post>
struct ConnectionRuleExpander<ConKind, Pre, Post, TypeList<>> {
    using type = TypeList<>;
};

template <template <typename, typename, typename> class ConKind,
          typename Pre, typename Post,
          typename RuleBase, typename... RuleRest>
struct ConnectionRuleExpander<ConKind, Pre, Post, TypeList<RuleBase, RuleRest...>> {
    using BareCon = ConKind<Pre, Post, TypePack<>>;
    using ThisRule = ConnectionRule<RuleBase, Pre, Post, BareCon>;
    using RecurseRules = ConnectionRuleExpander<ConKind, Pre, Post, TypeList<RuleRest...>>::type;
    using type = Append<TypeList<ThisRule>, RecurseRules>::type;
};



#endif //RULE_EXPANDER_H
