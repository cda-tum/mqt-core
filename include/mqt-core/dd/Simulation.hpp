/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/Operations.hpp"
#include "dd/Package_fwd.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/OpType.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <utility>

namespace dd {
using namespace qc;

template <class Config>
VectorDD simulate(const QuantumComputation& qc, const VectorDD& in,
                  Package<Config>& dd) {
  auto permutation = qc.initialLayout;
  auto e = in;
  for (const auto& op : qc) {
    // SWAP gates can be executed virtually by changing the permutation
    if (op->getType() == SWAP && !op->isControlled()) {
      const auto& targets = op->getTargets();
      std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
      continue;
    }

    e = applyUnitaryOperation(*op, e, dd, permutation);
  }
  changePermutation(e, permutation, qc.outputPermutation, dd);
  e = dd.reduceGarbage(e, qc.getGarbage());
  return e;
}

template <class Config>
std::map<std::string, std::size_t>
sample(const QuantumComputation& qc, const VectorDD& in, Package<Config>& dd,
       std::size_t shots, std::size_t seed = 0U);

/**
 * Sample from the output distribution of a quantum computation
 *
 * This method classically simulates the quantum computation @p qc and samples
 * @p shots times from the output distribution. The seed for the random number
 * generator can be set using the @p seed parameter.
 *
 * @param qc The quantum computation to simulate
 * @param shots The number of shots to sample
 * @param seed The seed for the random number generator
 * @return A histogram of the measurement results
 */
std::map<std::string, std::size_t> sample(const QuantumComputation& qc,
                                          std::size_t shots = 1024U,
                                          std::size_t seed = 0U);

template <class Config>
void extractProbabilityVector(const QuantumComputation& qc, const VectorDD& in,
                              SparsePVec& probVector, Package<Config>& dd);

template <class Config>
void extractProbabilityVectorRecursive(const QuantumComputation& qc,
                                       const VectorDD& currentState,
                                       decltype(qc.begin()) currentIt,
                                       Permutation& permutation,
                                       std::map<std::size_t, char> measurements,
                                       fp commonFactor, SparsePVec& probVector,
                                       Package<Config>& dd);
} // namespace dd
