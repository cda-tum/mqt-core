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
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Dialect/QTensor/Transforms/Passes.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir::qtensor {

#define GEN_PASS_DEF_NORMALIZEQTENSORPASS
#include "mqt/Dialect/QTensor/Transforms/Passes.h.inc"

static Value getTensorInput(Operation* operation) {
  if (auto extract = dyn_cast<ExtractOp>(operation)) {
    return extract.getTensor();
  }
  return cast<InsertOp>(operation).getDest();
}

static Value getTensorOutput(Operation* operation) {
  if (auto extract = dyn_cast<ExtractOp>(operation)) {
    return extract.getOutTensor();
  }
  return cast<InsertOp>(operation).getResult();
}

static std::optional<int64_t>
getAccessIndex(Operation* operation, DenseMap<Value, Attribute>& cache) {
  Value index;
  if (auto extract = dyn_cast<ExtractOp>(operation)) {
    index = extract.getIndex();
  } else if (auto insert = dyn_cast<InsertOp>(operation)) {
    index = insert.getIndex();
  } else {
    return std::nullopt;
  }
  const auto folded = mqt::valueToConstantAttr(index, cache);
  return folded ? getConstantIntValue(folded) : std::nullopt;
}

static bool normalizeChain(ArrayRef<Operation*> chain, IRRewriter& rewriter,
                           DenseMap<Value, Attribute>& cache) {
  bool sawInsert = false;
  bool needsReordering = false;
  for (auto* operation : chain) {
    if (isa<InsertOp>(operation)) {
      sawInsert = true;
    } else {
      needsReordering |= sawInsert;
    }
  }
  if (!needsReordering) {
    return false;
  }

  auto tensor = getTensorInput(chain.front());
  DenseMap<int64_t, SmallVector<size_t, 1>> pending;
  SmallVector<InsertOp> inserts;
  const auto flush = [&] {
    for (auto insert : inserts) {
      if (!insert) {
        continue;
      }
      auto replacement = cast<InsertOp>(rewriter.clone(*insert));
      replacement.getDestMutable().assign(tensor);
      tensor = replacement.getResult();
    }
    inserts.clear();
    pending.clear();
  };

  for (auto* operation : chain) {
    rewriter.setInsertionPoint(operation);
    const auto index = *getAccessIndex(operation, cache);
    if (auto insert = dyn_cast<InsertOp>(operation)) {
      // Keep overwritten writes: each insert still consumes its scalar.
      pending[index].push_back(inserts.size());
      inserts.push_back(insert);
      continue;
    }

    auto extract = cast<ExtractOp>(operation);
    Value qubit;
    if (auto found = pending.find(index); found != pending.end()) {
      auto& positions = found->second;
      auto& insert = inserts[positions.pop_back_val()];
      qubit = insert.getScalar();
      insert = nullptr;
      if (positions.empty()) {
        pending.erase(found);
      }
    } else {
      auto replacement = cast<ExtractOp>(rewriter.clone(*extract));
      replacement.getTensorMutable().assign(tensor);
      tensor = replacement.getOutTensor();
      qubit = replacement.getResult();
    }
    rewriter.replaceAllUsesWith(extract.getResult(), qubit);
  }

  rewriter.setInsertionPointAfter(chain.back());
  flush();
  rewriter.replaceAllUsesWith(getTensorOutput(chain.back()), tensor);
  for (auto* operation : llvm::reverse(chain)) {
    rewriter.eraseOp(operation);
  }
  return true;
}

static bool foldLocalRoundTrips(Block* block, IRRewriter& rewriter,
                                DenseMap<Value, Attribute>& cache) {
  bool changed = false;
  for (auto insert : llvm::make_early_inc_range(block->getOps<InsertOp>())) {
    auto extract = insert.getScalar().getDefiningOp<ExtractOp>();
    if (!extract || extract->getBlock() != block ||
        insert.getDest() != extract.getOutTensor()) {
      continue;
    }
    const auto insertIndex = getAccessIndex(insert, cache);
    const auto extractIndex = getAccessIndex(extract, cache);
    if (insert.getIndex() != extract.getIndex() &&
        (!insertIndex || !extractIndex || *insertIndex != *extractIndex)) {
      continue;
    }
    rewriter.replaceOp(insert, extract.getTensor());
    rewriter.eraseOp(extract);
    changed = true;
  }
  return changed;
}

static bool normalizeBlock(Block* block, IRRewriter& rewriter) {
  if (mayBeGraphRegion(*block->getParent())) {
    return false;
  }
  DenseMap<Value, Attribute> cache;
  // Reordering must not separate a local identity before it can fold.
  bool changed = foldLocalRoundTrips(block, rewriter, cache);
  SmallVector<Operation*> accesses;
  for (auto& operation : *block) {
    if (isa<ExtractOp, InsertOp>(operation)) {
      accesses.push_back(&operation);
    }
  }

  SmallPtrSet<Operation*, 32> visited;
  for (auto* access : accesses) {
    if (visited.contains(access)) {
      continue;
    }
    SmallVector<Operation*> chain;
    auto* current = access;
    while (current->getBlock() == block &&
           getAccessIndex(current, cache).has_value()) {
      visited.insert(current);
      chain.push_back(current);
      current = *getTensorOutput(current).getUsers().begin();
    }
    if (!chain.empty()) {
      changed |= normalizeChain(chain, rewriter, cache);
    }
  }
  return foldLocalRoundTrips(block, rewriter, cache) || changed;
}

namespace {
struct NormalizeQTensorPass final
    : impl::NormalizeQTensorPassBase<NormalizeQTensorPass> {
  using NormalizeQTensorPassBase::NormalizeQTensorPassBase;

protected:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    bool changed = false;
    getOperation().walk(
        [&](Block* block) { changed |= normalizeBlock(block, rewriter); });
    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};
} // namespace

} // namespace mlir::qtensor
