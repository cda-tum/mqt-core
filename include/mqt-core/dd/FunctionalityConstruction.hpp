/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/Package_fwd.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/OpType.hpp"

#include <cstddef>
#include <stack>

namespace dd {
using namespace qc;

/**
 * @brief Sequentially build a decision diagram representation for the
 * functionality of a purely-quantum @ref qc::QuantumComputation.
 *
 * @details For a circuit \f$G\f$ with \f$|G|\f$ gates
 * \f$g_0, g_1, \ldots, g_{|G|-1}\f$, the functionality of \f$G\f$ is defined as
 * the unitary matrix \f$U\f$ such that
 * \f[
 * U = U_{|G|-1}) \cdot U_{|G|-2} \cdot \ldots \cdot U_1 \cdot U_0,
 * \f]
 * where \f$U_i\f$ is the unitary matrix corresponding to gate \f$g_i\f$.
 * For an \f$n\f$-qubit quantum computation, \f$U\f$ is a \f$2^n \times 2^n\f$
 * matrix.
 *
 * By representing every single operation in the circuit as a decision diagram
 * instead of a unitary matrix and performing the matrix multiplication directly
 * using decision diagrams, a representation of the functionality of a quantum
 * computation can oftentimes be computed more efficiently in terms of memory
 * and runtime.
 *
 * This function effectively computes
 * \f[
 * DD(U) = DD(g_{|G|-1}) \otimes DD(g_{|G|-2}) \otimes \ldots \otimes DD(g_0)
 * \f]
 * by sequentially applying the decision diagrams of the gates in the circuit to
 * the current decision diagram representing the functionality of the quantum
 * computation.
 *
 * @param qc The quantum computation to construct the functionality for
 * @param dd The DD package to use for the construction
 * @tparam Config The configuration of the DD package
 * @return The matrix diagram representing the functionality of the quantum
 * computation
 */
template <class Config>
MatrixDD buildFunctionality(const QuantumComputation& qc, Package<Config>& dd);

/**
 * @brief Recursively build a decision diagram representation for the
 * functionality of a purely-quantum @ref qc::QuantumComputation.
 *
 * @see buildFunctionality
 * @details Instead of sequentially applying the decision diagrams of the gates
 * in the circuit, this function builds a binary computation tree out of the
 * decision diagrams of the gates in the circuit.
 * This results in a recursive pairwise grouping that can be more memory and
 * runtime efficient compared to the sequential approach.
 * @see https://arxiv.org/abs/2103.08281
 *
 * @param qc The quantum computation to construct the functionality for
 * @param dd The DD package to use for the construction
 * @tparam Config The configuration of the DD package
 * @return The matrix diagram representing the functionality of the quantum
 * computation
 */
template <class Config>
MatrixDD buildFunctionalityRecursive(const QuantumComputation& qc,
                                     Package<Config>& dd);

///-----------------------------------------------------------------------------
///                      \n Method Definitions \n
///-----------------------------------------------------------------------------

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
  const std::size_t leftIdx =
      opIdx & ~(static_cast<std::size_t>(1U) << (depth - 1U));
  if (!buildFunctionalityRecursive(qc, depth - 1U, leftIdx, s, permutation,
                                   dd)) {
    return false;
  }

  const std::size_t rightIdx =
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

} // namespace dd
