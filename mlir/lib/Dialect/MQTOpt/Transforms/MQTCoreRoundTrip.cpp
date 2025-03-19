/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/QuantumComputation.hpp"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/Operation.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_MQTCOREROUNDTRIP
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

struct MQTCoreRoundTrip final : impl::MQTCoreRoundTripBase<MQTCoreRoundTrip> {

  qc::QuantumComputation circuit;

  void runOnOperation() override {
    // Get the current operation being operated on.
    mlir::Operation* op = getOperation();
    mlir::MLIRContext* ctx = &getContext();

    // Define the set of patterns to use.
    mlir::RewritePatternSet patterns(ctx);
    populateToQuantumComputationPatterns(patterns, circuit);
    populateFromQuantumComputationPatterns(patterns, circuit);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
