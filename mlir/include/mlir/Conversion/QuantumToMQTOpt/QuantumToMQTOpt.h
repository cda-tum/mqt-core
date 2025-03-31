/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Pass/Pass.h" // from @llvm-project

namespace mlir::mqt::ir::conversions {

#define GEN_PASS_DECL
#include "mlir/Conversion/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h.inc"

} // namespace mlir::mqt::ir::conversions

#endif // LIB_CONVERSION_QUANTUMTOMQTOPT_QUANTUMTOMQTOPT_H_
