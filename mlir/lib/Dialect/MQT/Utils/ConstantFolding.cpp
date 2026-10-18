/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace mlir::mqt {

std::optional<double> attributeToDouble(Attribute attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValueAsDouble();
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    const bool isSigned = !intAttr.getType().isUnsignedInteger();
    APFloat value(APFloat::IEEEdouble(), APInt::getZero(64));
    value.convertFromAPInt(intAttr.getValue(), isSigned,
                           APFloat::rmNearestTiesToEven);
    return value.convertToDouble();
  }
  return std::nullopt;
}

std::optional<double> valueToDouble(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return std::nullopt;
  }
  return attributeToDouble(attr);
}

std::optional<Attribute>
valueToConstantAttr(Value value,
                    DenseMap<Value, std::optional<Attribute>>& cache) {
  SmallVector<std::pair<Value, unsigned>> worklist{{value, 0}};
  while (!worklist.empty()) {
    auto [current, nextOperand] = worklist.back();
    Operation* operation = current.getDefiningOp();
    if (nextOperand == 0) {
      auto [entry, inserted] = cache.try_emplace(current, std::nullopt);
      if (!inserted) {
        worklist.pop_back();
        continue;
      }
      Attribute attr;
      if (matchPattern(current, m_Constant(&attr))) {
        entry->second = attr;
        worklist.pop_back();
        continue;
      }
      if (operation == nullptr || operation->getNumRegions() != 0 ||
          !isPure(operation)) {
        worklist.pop_back();
        continue;
      }
      // Zero marks an unvisited value; operands use one-based indices.
      worklist.back().second = nextOperand = 1;
    }
    if (nextOperand <= operation->getNumOperands()) {
      Value operand = operation->getOperand(nextOperand - 1);
      const auto found = cache.find(operand);
      if (found == cache.end()) {
        worklist.emplace_back(operand, 0);
      } else if (!found->second) {
        // One dynamic operand is enough to rule out a constant expression.
        worklist.pop_back();
      } else {
        ++worklist.back().second;
      }
      continue;
    }

    SmallVector<Attribute> operands;
    for (Value operand : operation->getOperands()) {
      operands.push_back(*cache.lookup(operand));
    }
    SmallVector<OpFoldResult, 1> results;
    if (succeeded(operation->fold(operands, results)) && results.size() == 1) {
      if (auto attr = dyn_cast_if_present<Attribute>(results.front())) {
        cache[current] = attr;
      } else if (auto replacement =
                     dyn_cast_if_present<Value>(results.front())) {
        // Identity folds can return an existing value outside the operands.
        if (!cache.contains(replacement)) {
          worklist.emplace_back(replacement, 0);
          continue;
        }
        cache[current] = cache.lookup(replacement);
      }
    }
    worklist.pop_back();
  }
  return cache.lookup(value);
}

std::optional<Attribute> valueToConstantAttr(Value value) {
  if (isa<BlockArgument>(value)) {
    return std::nullopt;
  }
  Attribute literal;
  if (matchPattern(value, m_Constant(&literal))) {
    return literal;
  }
  DenseMap<Value, std::optional<Attribute>> cache;
  return valueToConstantAttr(value, cache);
}

std::optional<double> valueToConstantDouble(Value value) {
  if (const auto attr = valueToConstantAttr(value)) {
    return attributeToDouble(*attr);
  }
  return std::nullopt;
}

} // namespace mlir::mqt
