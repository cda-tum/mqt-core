// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QTENSOR_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_QTENSOR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ShrinkQTensorToFitPass : Pass<"qtensor-shrink-to-fit", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qtensor::QTensorDialect",
                           "mlir::arith::ArithDialect"];
  let summary = "Shrink static qtensors to their actually accessed indices.";
  let description = [{
    Shrinks one-dimensional static qtensors by tracing linear tensor chains from
    `qtensor.dealloc` to `qtensor.alloc` and rebuilding the chain on a compact
    allocation that only keeps accessed indices.
  }];
}

def NormalizeQTensorPass : Pass<"qtensor-normalize", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qtensor::QTensorDialect"];
  let summary = "Normalize constant-index QTensor access chains";
  let description = [{
    Folds local extract/reinsert identities, then reorders constant-index
    extracts before pending inserts within same-block SSA tensor chains. An
    extract of a pending inserted element uses that scalar directly. Pending
    inserts are emitted in their original order.
    Repeated writes are retained; an extract consumes only the latest pending
    write to its index, so overwritten qubits remain consumed.

    Unknown indices, unknown tensor users, and block boundaries terminate a
    chain. Constant-index segments beyond such boundaries are processed
    independently. Graph regions are skipped. Already batched chains are left
    unchanged.
    Retained accesses preserve their attributes and locations. The pass walks
    each access a bounded number of times and never commutes pairs iteratively.

    Pure index expressions are evaluated with a constant-folding cache; the
    pass does not rewrite those expressions. QCO cleanup runs reset analysis
    and canonicalization after this pass. Existing tensor canonicalizers handle
    accesses exposed later by control-flow simplification. This pass does not
    cross control-flow boundaries or optimize quantum gates.
  }];
}

#endif // MLIR_DIALECT_QTENSOR_TRANSFORMS_PASSES_TD
