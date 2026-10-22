/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <optional>

namespace mlir::mqt {

/// Require literal bounds before expanding a counted loop.
inline std::optional<llvm::APInt> getExactConstantTripCount(scf::ForOp loop) {
  if (!getConstantAPIntValue(getAsOpFoldResult(loop.getLowerBound())) ||
      !getConstantAPIntValue(getAsOpFoldResult(loop.getUpperBound())) ||
      !getConstantAPIntValue(getAsOpFoldResult(loop.getStep()))) {
    return std::nullopt;
  }
  return loop.getStaticTripCount();
}

/// Check the integer arithmetic used by MLIR's full-loop unroller.
inline bool canUseFullUnroll(scf::ForOp loop, uint64_t iterations) {
  const auto constant = [&](Value value) -> std::optional<int64_t> {
    const auto result = getConstantAPIntValue(getAsOpFoldResult(value));
    if (!result) {
      return std::nullopt;
    }
    if (loop.getUnsignedCmp()) {
      if (result->first.getActiveBits() > 63) {
        return std::nullopt;
      }
      return static_cast<int64_t>(result->first.getZExtValue());
    }
    return result->first.trySExtValue();
  };
  const auto lowerBound = constant(loop.getLowerBound());
  const auto upperBound = constant(loop.getUpperBound());
  const auto step = constant(loop.getStep());
  if (!lowerBound || !upperBound || !step || *step <= 0) {
    return false;
  }
  int64_t scaledStep = 0;
  int64_t unrolledUpperBound = 0;
  if (llvm::MulOverflow(*step, static_cast<int64_t>(iterations), scaledStep) !=
          0 ||
      llvm::AddOverflow(*lowerBound, scaledStep, unrolledUpperBound) != 0) {
    return false;
  }
  auto type = dyn_cast<IntegerType>(loop.getInductionVar().getType());
  return !type || (loop.getUnsignedCmp()
                       ? llvm::isUIntN(type.getWidth(),
                                       static_cast<uint64_t>(scaledStep))
                       : llvm::isIntN(type.getWidth(), scaledStep));
}

} // namespace mlir::mqt
