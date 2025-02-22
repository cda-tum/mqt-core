/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Definitions.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"
#include "ir/Permutation.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dd {

template <class Config>
qc::MatrixDD getStandardOperationDD(Package<Config>& dd, qc::OpType type,
                                    const std::vector<fp>& params,
                                    const qc::Controls& controls,
                                    const std::vector<qc::Qubit>& targets) {
  if (qc::isSingleQubitGate(type)) {
    if (targets.size() != 1) {
      throw qc::QFRException(
          "Expected exactly one target qubit for single-qubit gate");
    }
    return dd.makeGateDD(opToSingleQubitGateMatrix(type, params), controls,
                         targets[0U]);
  }
  if (qc::isTwoQubitGate(type)) {
    if (targets.size() != 2) {
      throw qc::QFRException("Expected two target qubits for two-qubit gate");
    }
    return dd.makeTwoQubitGateDD(opToTwoQubitGateMatrix(type, params), controls,
                                 targets[0U], targets[1U]);
  }
  throw qc::QFRException("Unexpected operation type");
}

template <class Config>
qc::MatrixDD getStandardOperationDD(const qc::StandardOperation& op,
                                    Package<Config>& dd,
                                    const qc::Controls& controls,
                                    const std::vector<qc::Qubit>& targets,
                                    const bool inverse) {
  auto type = op.getType();

  if (!inverse) {
    return getStandardOperationDD(dd, type, op.getParameter(), controls,
                                  targets);
  }

  // invert the operation
  std::vector<fp> params = op.getParameter();
  std::vector<qc::Qubit> targetQubits = targets;

  switch (type) {
  // operations that are self-inverse do not need any changes
  case qc::I:
  case qc::H:
  case qc::X:
  case qc::Y:
  case qc::Z:
  case qc::SWAP:
  case qc::ECR:
    break;
  // operations that have an inverse gate with the same parameters
  case qc::iSWAP:
  case qc::iSWAPdg:
  case qc::Peres:
  case qc::Peresdg:
  case qc::S:
  case qc::Sdg:
  case qc::T:
  case qc::Tdg:
  case qc::V:
  case qc::Vdg:
  case qc::SX:
  case qc::SXdg:
    type = static_cast<qc::OpType>(+type ^ qc::OpTypeInv);
    break;
  // operations that can be inversed by negating the first parameter
  case qc::RXX:
  case qc::RYY:
  case qc::RZZ:
  case qc::RZX:
  case qc::RX:
  case qc::RY:
  case qc::RZ:
  case qc::P:
  case qc::XXminusYY:
  case qc::XXplusYY:
    params[0U] = -params[0U];
    break;
  // other special cases
  case qc::DCX:
    if (targetQubits.size() != 2) {
      throw qc::QFRException("Invalid target qubits for DCX");
    }
    // DCX is not self-inverse, but the inverse is just swapping the targets
    std::swap(targetQubits[0], targetQubits[1]);
    break;
  // invert all parameters
  case qc::U:
    // swap [a, b, c] to [a, c, b]
    std::swap(params[1U], params[2U]);
    for (auto& param : params) {
      param = -param;
    }
    break;
  case qc::U2:
    std::swap(params[0U], params[1U]);
    params[0U] = -params[0U] - PI;
    params[1U] = -params[1U] + PI;
    break;

  default:
    std::ostringstream oss{};
    oss << "negation for gate " << op.getName() << " not available!";
    throw qc::QFRException(oss.str());
  }
  return getStandardOperationDD(dd, type, params, controls, targetQubits);
}

// The methods with a permutation parameter apply these Operations according to
// the mapping specified by the permutation, e.g.
//      if perm[0] = 1 and perm[1] = 0
//      then cx 0 1 will be translated to cx perm[0] perm[1] == cx 1 0
// An empty permutation marks the identity permutation.

template <class Config>
qc::MatrixDD getDD(const qc::Operation& op, Package<Config>& dd,
                   const qc::Permutation& permutation = {},
                   const bool inverse = false) {
  const auto type = op.getType();

  if (type == qc::Barrier) {
    return dd.makeIdent();
  }

  if (type == qc::GPhase) {
    auto phase = op.getParameter()[0U];
    if (inverse) {
      phase = -phase;
    }
    auto id = dd.makeIdent();
    id.w = dd.cn.lookup(std::cos(phase), std::sin(phase));
    return id;
  }

  if (op.isStandardOperation()) {
    const auto& standardOp = dynamic_cast<const qc::StandardOperation&>(op);
    const auto& targets = permutation.apply(standardOp.getTargets());
    const auto& controls = permutation.apply(standardOp.getControls());

    return getStandardOperationDD(standardOp, dd, controls, targets, inverse);
  }

  if (op.isCompoundOperation()) {
    const auto& compoundOp = dynamic_cast<const qc::CompoundOperation&>(op);
    auto e = dd.makeIdent();
    if (inverse) {
      for (const auto& operation : compoundOp) {
        e = dd.multiply(e, getInverseDD(*operation, dd, permutation));
      }
    } else {
      for (const auto& operation : compoundOp) {
        e = dd.multiply(getDD(*operation, dd, permutation), e);
      }
    }
    return e;
  }

  if (op.isClassicControlledOperation()) {
    const auto& classicOp =
        dynamic_cast<const qc::ClassicControlledOperation&>(op);
    return getDD(*classicOp.getOperation(), dd, permutation, inverse);
  }

  assert(op.isNonUnitaryOperation());
  throw qc::QFRException("DD for non-unitary operation not available!");
}

template <class Config>
qc::MatrixDD getInverseDD(const qc::Operation& op, Package<Config>& dd,
                          const qc::Permutation& permutation = {}) {
  return getDD(op, dd, permutation, true);
}

template <class Config, class Node>
Edge<Node> applyUnitaryOperation(const qc::Operation& op, const Edge<Node>& in,
                                 Package<Config>& dd,
                                 const qc::Permutation& permutation = {}) {
  static_assert(std::is_same_v<Node, dd::vNode> ||
                std::is_same_v<Node, dd::mNode>);
  return dd.applyOperation(getDD(op, dd, permutation), in);
}

template <class Config>
qc::VectorDD applyMeasurement(const qc::NonUnitaryOperation& op,
                              qc::VectorDD in, Package<Config>& dd,
                              std::mt19937_64& rng,
                              std::vector<bool>& measurements,
                              const qc::Permutation& permutation = {}) {
  assert(op.getType() == qc::Measure);
  const auto& qubits = permutation.apply(op.getTargets());
  const auto& bits = op.getClassics();
  for (size_t j = 0U; j < qubits.size(); ++j) {
    measurements.at(bits.at(j)) =
        dd.measureOneCollapsing(in, static_cast<dd::Qubit>(qubits.at(j)),
                                rng) == '1';
  }
  return in;
}

template <class Config>
qc::VectorDD applyReset(const qc::NonUnitaryOperation& op, qc::VectorDD in,
                        Package<Config>& dd, std::mt19937_64& rng,
                        const qc::Permutation& permutation = {}) {
  assert(op.getType() == qc::Reset);
  const auto& qubits = permutation.apply(op.getTargets());
  for (const auto& qubit : qubits) {
    const auto bit =
        dd.measureOneCollapsing(in, static_cast<dd::Qubit>(qubit), rng);
    // apply an X operation whenever the measured result is one
    if (bit == '1') {
      const auto x = qc::StandardOperation(qubit, qc::X);
      in = applyUnitaryOperation(x, in, dd);
    }
  }
  return in;
}

template <class Config>
qc::VectorDD
applyClassicControlledOperation(const qc::ClassicControlledOperation& op,
                                const qc::VectorDD& in, Package<Config>& dd,
                                std::vector<bool>& measurements,
                                const qc::Permutation& permutation = {}) {
  const auto& expectedValue = op.getExpectedValue();
  const auto& comparisonKind = op.getComparisonKind();

  // determine the actual value from measurements
  auto actualValue = 0ULL;
  if (const auto& controlRegister = op.getControlRegister();
      controlRegister.has_value()) {
    assert(!op.getControlBit().has_value());
    const auto regStart = controlRegister->getStartIndex();
    const auto regSize = controlRegister->getSize();
    for (std::size_t j = 0; j < regSize; ++j) {
      if (measurements[regStart + j]) {
        actualValue |= 1ULL << j;
      }
    }
  }
  if (const auto& controlBit = op.getControlBit(); controlBit.has_value()) {
    assert(!op.getControlRegister().has_value());
    actualValue = measurements[*controlBit] ? 1U : 0U;
  }

  // check if the actual value matches the expected value according to the
  // comparison kind
  const auto control = [actualValue, expectedValue, comparisonKind]() {
    switch (comparisonKind) {
    case qc::ComparisonKind::Eq:
      return actualValue == expectedValue;
    case qc::ComparisonKind::Neq:
      return actualValue != expectedValue;
    case qc::ComparisonKind::Lt:
      return actualValue < expectedValue;
    case qc::ComparisonKind::Leq:
      return actualValue <= expectedValue;
    case qc::ComparisonKind::Gt:
      return actualValue > expectedValue;
    case qc::ComparisonKind::Geq:
      return actualValue >= expectedValue;
    }
    qc::unreachable();
  }();

  if (!control) {
    return in;
  }

  return applyUnitaryOperation(op, in, dd, permutation);
}

// apply swaps 'on' DD in order to change 'from' to 'to'
// where |from| >= |to|
template <class DDType, class Config>
void changePermutation(DDType& on, qc::Permutation& from,
                       const qc::Permutation& to, Package<Config>& dd,
                       const bool regular = true) {
  assert(from.size() >= to.size());
  if (on.isZeroTerminal()) {
    return;
  }

  // iterate over (k,v) pairs of second permutation
  for (const auto& [i, goal] : to) {
    // search for key in the first map
    auto it = from.find(i);
    if (it == from.end()) {
      throw qc::QFRException(
          "[changePermutation] Key " + std::to_string(it->first) +
          " was not found in first permutation. This should never happen.");
    }
    auto current = it->second;

    // permutations agree for this key value
    if (current == goal) {
      continue;
    }

    // search for goal value in first permutation
    qc::Qubit j = 0;
    for (const auto& [key, value] : from) {
      if (value == goal) {
        j = key;
        break;
      }
    }

    // swap i and j
    auto saved = on;
    const auto swapDD = dd.makeTwoQubitGateDD(opToTwoQubitGateMatrix(qc::SWAP),
                                              from.at(i), from.at(j));
    if constexpr (std::is_same_v<DDType, qc::VectorDD>) {
      on = dd.multiply(swapDD, on);
    } else {
      // the regular flag only has an effect on matrix DDs
      if (regular) {
        on = dd.multiply(swapDD, on);
      } else {
        on = dd.multiply(on, swapDD);
      }
    }

    dd.incRef(on);
    dd.decRef(saved);
    dd.garbageCollect();

    // update permutation
    from.at(i) = goal;
    from.at(j) = current;
  }
}

} // namespace dd
