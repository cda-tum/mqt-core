/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/FunctionalityConstruction.hpp"

#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/OpType.hpp"

#include <cmath>
#include <cstddef>
#include <stack>
#include <utility>

namespace dd {
template <class Config>
MatrixDD buildFunctionality(const QuantumComputation& qc, Package<Config>& dd) {
  if (qc.getNqubits() == 0U) {
    return MatrixDD::one();
  }

  auto permutation = qc.initialLayout;
  auto e = dd.createInitialMatrix(qc.getAncillary());

  for (const auto& op : qc) {
    // SWAP gates can be executed virtually by changing the permutation
    if (op->getType() == OpType::SWAP && !op->isControlled()) {
      const auto& targets = op->getTargets();
      std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
      continue;
    }

    e = applyUnitaryOperation(*op, e, dd, permutation);
  }
  // correct permutation if necessary
  changePermutation(e, permutation, qc.outputPermutation, dd);
  e = dd.reduceAncillae(e, qc.getAncillary());
  e = dd.reduceGarbage(e, qc.getGarbage());

  return e;
}

namespace {
template <class Config>
bool buildFunctionalityRecursive(const QuantumComputation& qc,
                                 std::size_t depth, std::size_t opIdx,
                                 std::stack<MatrixDD>& s,
                                 Permutation& permutation,
                                 Package<Config>& dd) {
  // base case
  if (depth == 1U) {
    auto e = dd.makeIdent();
    if (const auto& op = qc.at(opIdx);
        op->getType() == OpType::SWAP && !op->isControlled()) {
      const auto& targets = op->getTargets();
      std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
    } else {
      e = getDD(*qc.at(opIdx), dd, permutation);
    }
    ++opIdx;
    if (opIdx == qc.size()) {
      // only one element was left
      s.push(e);
      dd.incRef(e);
      return false;
    }
    auto f = dd.makeIdent();
    if (const auto& op = qc.at(opIdx);
        op->getType() == OpType::SWAP && !op->isControlled()) {
      const auto& targets = op->getTargets();
      std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
    } else {
      f = getDD(*qc.at(opIdx), dd, permutation);
    }
    s.push(dd.multiply(f, e)); // ! reverse multiplication
    dd.incRef(s.top());
    return (opIdx != qc.size() - 1U);
  }

  // in case no operations are left after the first recursive call nothing has
  // to be done
  const size_t leftIdx =
      opIdx & ~(static_cast<std::size_t>(1U) << (depth - 1U));
  if (!buildFunctionalityRecursive(qc, depth - 1U, leftIdx, s, permutation,
                                   dd)) {
    return false;
  }

  const size_t rightIdx =
      opIdx | (static_cast<std::size_t>(1U) << (depth - 1U));
  const auto success =
      buildFunctionalityRecursive(qc, depth - 1U, rightIdx, s, permutation, dd);

  // get latest two results from stack and push their product on the stack
  auto e = s.top();
  s.pop();
  auto f = s.top();
  s.pop();
  s.push(dd.multiply(e, f)); // ordering because of stack structure

  // reference counting
  dd.decRef(e);
  dd.decRef(f);
  dd.incRef(s.top());
  dd.garbageCollect();

  return success;
}
} // namespace

template <class Config>
MatrixDD buildFunctionalityRecursive(const QuantumComputation& qc,
                                     Package<Config>& dd) {
  if (qc.getNqubits() == 0U) {
    return MatrixDD::one();
  }

  auto permutation = qc.initialLayout;

  if (qc.size() == 1U) {
    auto e = getDD(*qc.front(), dd, permutation);
    dd.incRef(e);
    return e;
  }

  std::stack<MatrixDD> s{};
  auto depth = static_cast<std::size_t>(std::ceil(std::log2(qc.size())));
  buildFunctionalityRecursive(qc, depth, 0, s, permutation, dd);
  auto e = s.top();
  s.pop();

  // correct permutation if necessary
  changePermutation(e, permutation, qc.outputPermutation, dd);
  e = dd.reduceAncillae(e, qc.getAncillary());
  e = dd.reduceGarbage(e, qc.getGarbage());

  return e;
}

template MatrixDD buildFunctionality(const QuantumComputation& qc,
                                     Package<>& dd);
template MatrixDD
buildFunctionality(const QuantumComputation& qc,
                   Package<DensityMatrixSimulatorDDPackageConfig>& dd);
template MatrixDD
buildFunctionality(const QuantumComputation& qc,
                   Package<StochasticNoiseSimulatorDDPackageConfig>& dd);

template MatrixDD buildFunctionality(const QuantumComputation& qc,
                                     UnitarySimulatorDDPackage& dd);

template MatrixDD buildFunctionalityRecursive(const QuantumComputation& qc,
                                              Package<>& dd);
template MatrixDD buildFunctionalityRecursive(const QuantumComputation& qc,
                                              UnitarySimulatorDDPackage& dd);

} // namespace dd
