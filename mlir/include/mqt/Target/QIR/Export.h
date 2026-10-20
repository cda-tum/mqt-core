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

#include "mlir/Support/LLVM.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
class ModuleOp;

namespace qir {

/// Export lowered QIR to verified LLVM IR with normalized QIR module flags.
///
/// Leaves the stream unchanged when translation or verification fails.
LogicalResult translateQIRToLLVMIR(ModuleOp moduleOp,
                                   llvm::raw_ostream& output);

/// Export lowered QIR to LLVM bitcode with normalized QIR module flags.
///
/// Leaves the stream unchanged when translation or verification fails.
LogicalResult translateQIRToBitcode(ModuleOp moduleOp,
                                    llvm::raw_ostream& output);

/// Write verified QIR bitcode, reporting translation and file errors.
LogicalResult writeQIRBitcode(ModuleOp moduleOp, StringRef filename);

} // namespace qir
} // namespace mlir
