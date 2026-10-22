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
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_REMOVEREDUNDANTRESETS
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {
struct TensorProvenance {
  bool knownIndices = true;
  DenseSet<int64_t> touched;
};
} // namespace

static bool removeRedundantResets(Block* block, IRRewriter& rewriter) {
  if (mayBeGraphRegion(*block->getParent())) {
    return false;
  }
  bool changed = false;
  DenseMap<Value, std::unique_ptr<TensorProvenance>> tensors;
  DenseSet<Value> zeroQubits;
  DenseMap<Value, Attribute> cache;
  const auto constantIndex = [&](Value index) {
    const auto folded = mqt::valueToConstantAttr(index, cache);
    return folded ? getConstantIntValue(folded) : std::nullopt;
  };
  for (auto& operation : llvm::make_early_inc_range(*block)) {
    if (auto alloc = dyn_cast<AllocOp>(operation)) {
      zeroQubits.insert(alloc.getResult());
    } else if (auto alloc = dyn_cast<qtensor::AllocOp>(operation)) {
      tensors[alloc.getResult()] = std::make_unique<TensorProvenance>();
    } else if (auto extract = dyn_cast<qtensor::ExtractOp>(operation)) {
      auto state = std::move(tensors[extract.getTensor()]);
      tensors.erase(extract.getTensor());
      if (!state) {
        continue;
      }
      if (auto index = constantIndex(extract.getIndex())) {
        if (state->knownIndices && !state->touched.contains(*index)) {
          zeroQubits.insert(extract.getResult());
        }
        state->touched.insert(*index);
      } else {
        state->knownIndices = false;
      }
      tensors[extract.getOutTensor()] = std::move(state);
    } else if (auto insert = dyn_cast<qtensor::InsertOp>(operation)) {
      auto state = std::move(tensors[insert.getDest()]);
      tensors.erase(insert.getDest());
      if (!state) {
        continue;
      }
      if (auto index = constantIndex(insert.getIndex())) {
        state->touched.insert(*index);
      } else {
        state->knownIndices = false;
      }
      tensors[insert.getResult()] = std::move(state);
    } else if (auto reset = dyn_cast<ResetOp>(operation)) {
      if (zeroQubits.contains(reset.getQubitIn())) {
        rewriter.replaceOp(reset, reset.getQubitIn());
        changed = true;
      } else {
        zeroQubits.insert(reset.getQubitOut());
      }
    }
  }
  return changed;
}

namespace {
struct RemoveRedundantResets final
    : impl::RemoveRedundantResetsBase<RemoveRedundantResets> {
  using RemoveRedundantResetsBase::RemoveRedundantResetsBase;

protected:
  void runOnOperation() override {
    if (!getOperation()
             .walk([](ResetOp) { return WalkResult::interrupt(); })
             .wasInterrupted()) {
      markAllAnalysesPreserved();
      return;
    }
    IRRewriter rewriter(&getContext());
    bool changed = false;
    getOperation().walk([&](Block* block) {
      changed |= removeRedundantResets(block, rewriter);
    });
    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};
} // namespace

} // namespace mlir::qco
