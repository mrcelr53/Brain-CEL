/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */
#ifndef BUILD_H
#define BUILD_H

#include "../../src/core/meta/type_pack.h"
#include "../../src/core/meta/type_list.h"
#include "../../src/core/meta/con_maker.h"

#include "../../src/functional/Variable.h"
#include "../../src/functional/Function.h"

#include "../../src/spiking/Axon.h"
#include "../../src/spiking/Soma.h"
#include "../../src/spiking/Dendrite.h"
#include "../../src/spiking/Neuron.h"
#include "../../src/spiking/Synapse.h"
#include "../../src/spiking/Plasticity.h"
#include "../../src/spiking/Membrane.h"
#include "../../src/spiking/Layer.h"


// Typed Input & Output & Parameter
template <typename ParameterPack>
using F32Input = TypedInput<ParameterPack, float>;
template <typename ParameterPack>
using F32Output = TypedOutput<ParameterPack, float>;
template <typename ParameterPack>
using VecF32Input = TypedInput<ParameterPack, std::vector<float>>;
template <typename ParameterPack>
using VecF32Output = TypedOutput<ParameterPack, std::vector<float>>;

// Build all functional template types
struct VariablePack : ConnectionMaker<
    Variable,                                              // Connection type
    TerminalList<F32Output, VecF32Output>,   // Pre-terminals
    TerminalList<F32Input, VecF32Input>       // Post-terminals
>::Maker<VariablePack>::type {};
using F32VariableType              = PackElement_t<0, VariablePack>;
using F32VecF32VariableType        = PackElement_t<1, VariablePack>;
using VecF32F32VariableType        = PackElement_t<2, VariablePack>;
using VecF32VariableType           = PackElement_t<3, VariablePack>;

using F32InputType = TypedInput<VariablePack, float>;
using F32OutputType = TypedOutput<VariablePack, float>;
using VecF32InputType = TypedInput<VariablePack, std::vector<float>>;
using VecF32OutputType = TypedOutput<VariablePack, std::vector<float>>;

using InVariablePack = TypeVectorPack<F32InputType, VecF32InputType>;
using OutVariablePack = TypeVectorPack<F32OutputType, VecF32OutputType>;

using FunctionType = Function<InVariablePack, OutVariablePack, TypePack<>>;
using NodePack = TypeVectorPack<FunctionType>;



// Build all neuronal template types
struct SynapsePack : ConnectionMaker<
    Synapse,                             // Connection type
    TerminalList<Axon>,                  // Pre-terminals
    TerminalList<Soma, Dendrite>,        // Post-terminals
    TypeList<SparseSTDPf, SparseSTDP>,   // Process pack
    true                                 // Learning possible
>::Maker<SynapsePack>::type {};
using AxoSomaticStaticSynapseType     = PackElement_t<0, SynapsePack>;  // Axon -> Soma      static
using AxoSomaticSynapseType           = PackElement_t<1, SynapsePack>;  // Axon -> Soma      learning
using AxoDendriticStaticSynapseType   = PackElement_t<2, SynapsePack>;  // Axon -> Dendrite  static
using AxoDendriticSynapseType         = PackElement_t<3, SynapsePack>;  // Axon -> Dendrite  learning

using SomaType = Soma<SynapsePack>;
using AxonType = Axon<SynapsePack>;
using DendriteType = Dendrite<SynapsePack>;

using NeuronType = Neuron<SynapsePack, SynapsePack, InVariablePack, OutVariablePack, TypePack<LifMembrane, IzhMembrane>>;
using NeuronPack = TypeVectorPack<NeuronType>;


#endif //BUILD_H
