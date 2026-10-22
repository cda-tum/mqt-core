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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
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

/// Expand a constant counted loop within a cumulative cloning budget.
///
/// Requires literal bounds and their exact trip count. Diagnose unsupported
/// arithmetic or excessive expansion before changing the loop. Callers fold
/// the resulting constants and select any newly static nested loops.
inline LogicalResult unrollStaticLoop(scf::ForOp loop,
                                      const llvm::APInt& tripCount,
                                      uint64_t maxOperations,
                                      uint64_t& clonedOperations) {
  IRRewriter rewriter(loop.getContext());
  if (tripCount.ule(1)) {
    if (failed(loop.promoteIfSingleIteration(rewriter))) {
      return loop.emitError(
          "failed to simplify a zero- or single-iteration loop");
    }
    return success();
  }

  uint64_t bodyOperations = 0;
  Operation* terminator = loop.getBody()->getTerminator();
  loop.getRegion().walk(
      [&](Operation* operation) { bodyOperations += operation != terminator; });
  const bool emptyBody = bodyOperations == 0;
  bodyOperations = std::max(bodyOperations, uint64_t{1});
  if ((tripCount - 1)
          .ugt((maxOperations - clonedOperations) / bodyOperations)) {
    return loop.emitError() << "full unrolling would clone more than "
                            << maxOperations << " loop-body operations";
  }
  if (tripCount.getActiveBits() > 64 ||
      !canUseFullUnroll(loop, tripCount.getZExtValue())) {
    return loop.emitError(
        "cannot safely apply MLIR full unrolling to these loop bounds");
  }
  clonedOperations += bodyOperations * (tripCount.getZExtValue() - 1);
  if (emptyBody) {
    // LLVM skips terminator-only bodies; remove this constant when it supports
    // unrolling them. Caller cleanup removes it after remapping carried values.
    rewriter.setInsertionPointToStart(loop.getBody());
    arith::ConstantIndexOp::create(rewriter, loop.getLoc(), 0);
  }
  if (failed(loopUnrollFull(loop))) {
    return loop.emitError("failed to fully unroll a static counted loop");
  }
  return success();
}

} // namespace mlir::mqt
