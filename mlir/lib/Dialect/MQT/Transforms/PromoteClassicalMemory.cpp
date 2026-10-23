/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::mqt {

#define GEN_PASS_DEF_PROMOTECLASSICALMEMORY
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"

namespace {
struct PromoteClassicalMemory final
    : impl::PromoteClassicalMemoryBase<PromoteClassicalMemory> {
  using PromoteClassicalMemoryBase::PromoteClassicalMemoryBase;
  OpPassManager pipeline_{ModuleOp::getOperationName()};

  PromoteClassicalMemory() {
    // Fold index normalization before upstream promotion inspects accesses.
    pipeline_.addPass(createCanonicalizerPass());
    pipeline_.addPass(createSROA());
    pipeline_.addPass(createMem2Reg());
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    pipeline_.getDependentDialects(registry);
  }

protected:
  void runOnOperation() override {
    const auto allocation = getOperation().walk([](Operation* op) {
      return isa<PromotableAllocationOpInterface,
                 DestructurableAllocationOpInterface>(op)
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (!allocation.wasInterrupted()) {
      markAllAnalysesPreserved();
      return;
    }
    if (failed(runPipeline(pipeline_, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::mqt
