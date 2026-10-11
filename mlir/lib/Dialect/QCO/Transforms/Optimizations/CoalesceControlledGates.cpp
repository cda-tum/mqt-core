/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_COALESCECONTROLLEDGATES
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

struct CoalesceControls final : OpRewritePattern<CtrlOp> {
  CoalesceControls(MLIRContext* context, const CompilerTarget* target)
      : OpRewritePattern(context), target_(target) {}

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() == 0) {
      return failure();
    }
    // Cross only scalar computations; barriers and other quantum operations
    // retain their position even when they act on disjoint wires.
    auto* candidate = op->getNextNode();
    while (candidate != nullptr && candidate->getNumRegions() == 0 &&
           !isa<UnitaryOpInterface>(candidate) &&
           isMemoryEffectFree(candidate) &&
           !llvm::any_of(candidate->getOperandTypes(), isLinearQubitType) &&
           !llvm::any_of(candidate->getResultTypes(), isLinearQubitType)) {
      candidate = candidate->getNextNode();
    }
    auto next = dyn_cast_or_null<CtrlOp>(candidate);
    if (!next || op.getNumControls() != next.getNumControls() ||
        !llvm::equal(op.getResults(), next.getOperands())) {
      return failure();
    }
    if (target_ != nullptr) {
      if (target_->supports(next)) {
        return failure();
      }
      for (auto unitary = cast<UnitaryOpInterface>(op.getOperation()); unitary;
           unitary = unitary->getParentOfType<UnitaryOpInterface>()) {
        if (target_->supports(unitary)) {
          return failure();
        }
      }
    }

    // Move the existing shell to retain parameter dominance without moving
    // the accumulated body again for each member of a long control chain.
    rewriter.moveOpBefore(op, next);
    rewriter.modifyOpInPlace(op, [&] {
      auto* yield = op.getBody()->getTerminator();
      SmallVector<Value> values(yield->getOperands());
      rewriter.inlineBlockBefore(next.getBody(), yield, values);
      rewriter.eraseOp(yield);
    });
    rewriter.replaceOp(next, op.getResults());
    return success();
  }

private:
  const CompilerTarget* target_;
};

struct CoalesceControlledGates final
    : impl::CoalesceControlledGatesBase<CoalesceControlledGates> {
  using Base::Base;

  CoalesceControlledGates() = default;

  explicit CoalesceControlledGates(const CompilerTarget& target)
      : target_(target) {}

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CoalesceControls>(&getContext(),
                                   target_ ? &*target_ : nullptr);
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns),
            GreedyRewriteConfig{}.setUseTopDownTraversal()))) {
      signalPassFailure();
    }
  }

private:
  std::optional<CompilerTarget> target_;
};

} // namespace

std::unique_ptr<Pass>
createCoalesceControlledGates(const CompilerTarget& target) {
  return std::make_unique<CoalesceControlledGates>(target);
}

} // namespace mlir::qco
