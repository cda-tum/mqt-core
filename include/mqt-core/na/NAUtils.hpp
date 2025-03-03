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
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <utility>

namespace na {
/**
 * @brief Checks whether a gate is global.
 * @details A StandardOperation is global if it acts on all qubits.
 * A CompoundOperation is global if all its sub-operations are
 * StandardOperations of the same type with the same parameters acting on all
 * qubits. The latter is what a QASM line like `ry(π) q;` is translated to in
 * MQT-core. All other operations are not global.
 */
[[nodiscard]] inline auto isGlobal(const qc::Operation& op,
                                   const std::size_t nQubits) -> bool {
  if (op.isStandardOperation()) {
    return op.getUsedQubits().size() == nQubits;
  }
  if (op.isCompoundOperation()) {
    const auto ops = dynamic_cast<const qc::CompoundOperation&>(op);
    const auto& params = ops.at(0)->getParameter();
    const auto& type = ops.at(0)->getType();
    return op.getUsedQubits().size() == nQubits &&
           std::all_of(ops.cbegin(), ops.cend(), [&](const auto& operation) {
             return operation->isStandardOperation() &&
                    operation->getNcontrols() == 0 &&
                    operation->getType() == type &&
                    operation->getParameter() == params;
           });
  }
  return false;
}
} // namespace na

template <> struct std::hash<std::pair<qc::OpType, std::size_t>> {
  std::size_t
  operator()(const std::pair<qc::OpType, std::size_t>& t) const noexcept {
    const std::size_t h1 = std::hash<qc::OpType>{}(t.first);
    const std::size_t h2 = std::hash<std::size_t>{}(t.second);
    return qc::combineHash(h1, h2);
  }
};

template <> struct std::hash<std::pair<std::size_t, std::size_t>> {
  std::size_t
  operator()(const std::pair<std::size_t, std::size_t>& p) const noexcept {
    const std::size_t h1 = std::hash<std::size_t>{}(p.first);
    const std::size_t h2 = std::hash<std::size_t>{}(p.second);
    return qc::combineHash(h1, h2);
  }
};
