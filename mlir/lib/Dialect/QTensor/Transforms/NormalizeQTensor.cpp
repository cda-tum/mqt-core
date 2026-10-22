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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
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

struct NormalizationAccess {
  Operation* operation;
  int64_t index;
  bool erase = false;
};

static bool normalizeChain(MutableArrayRef<NormalizationAccess> chain,
                           IRRewriter& rewriter) {
  bool sawInsert = false;
  bool needsReordering = false;
  for (auto& access : chain) {
    if (isa<InsertOp>(access.operation)) {
      sawInsert = true;
    } else {
      needsReordering |= sawInsert;
    }
  }
  if (!needsReordering) {
    return false;
  }

  auto tensor = getTensorInput(chain.front().operation);
  auto* block = chain.back().operation->getBlock();
  auto insertionPoint = std::next(chain.back().operation->getIterator());
  auto& outputUse = *getTensorOutput(chain.back().operation).use_begin();
  DenseMap<int64_t, SmallVector<size_t, 1>> pending;

  for (auto [position, access] : llvm::enumerate(chain)) {
    if (isa<InsertOp>(access.operation)) {
      // Keep overwritten writes: each insert still consumes its scalar.
      pending[access.index].push_back(position);
      continue;
    }

    auto extract = cast<ExtractOp>(access.operation);
    if (auto found = pending.find(access.index); found != pending.end()) {
      auto& positions = found->second;
      auto& stored = chain[positions.pop_back_val()];
      auto insert = cast<InsertOp>(stored.operation);
      rewriter.replaceAllUsesWith(extract.getResult(), insert.getScalar());
      stored.erase = true;
      access.erase = true;
      if (positions.empty()) {
        pending.erase(found);
      }
    } else {
      rewriter.modifyOpInPlace(
          extract, [&] { extract.getTensorMutable().assign(tensor); });
      tensor = extract.getOutTensor();
    }
  }

  for (auto& access : chain) {
    auto insert = dyn_cast<InsertOp>(access.operation);
    if (!insert || access.erase) {
      continue;
    }
    rewriter.moveOpBefore(insert, block, insertionPoint);
    rewriter.modifyOpInPlace(insert,
                             [&] { insert.getDestMutable().assign(tensor); });
    tensor = insert.getResult();
  }
  rewriter.modifyOpInPlace(outputUse.getOwner(),
                           [&] { outputUse.set(tensor); });
  // Dead tensor results only feed other dead accesses; avoid rewiring them.
  for (auto& access : llvm::reverse(chain)) {
    if (access.erase) {
      rewriter.eraseOp(access.operation);
    }
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
    if (insert.getIndex() != extract.getIndex()) {
      const auto insertIndex = getAccessIndex(insert, cache);
      const auto extractIndex = getAccessIndex(extract, cache);
      if (!insertIndex || !extractIndex || *insertIndex != *extractIndex) {
        continue;
      }
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
    SmallVector<NormalizationAccess> chain;
    auto* current = access;
    while (current->getBlock() == block) {
      const auto index = getAccessIndex(current, cache);
      if (!index) {
        break;
      }
      visited.insert(current);
      chain.push_back({current, *index});
      current = *getTensorOutput(current).getUsers().begin();
    }
    if (!chain.empty()) {
      changed |= normalizeChain(chain, rewriter);
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
