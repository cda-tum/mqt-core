/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef LIB_CONVERSION_MQTOPTTOQUANTUM_MQTOPTTOQUANTUM_H_
#define LIB_CONVERSION_MQTOPTTOQUANTUM_MQTOPTTOQUANTUM_H_

#include "mlir/Pass/Pass.h" // from @llvm-project

namespace mlir::mqt::ir::conversions {

#define GEN_PASS_DECL
#include "mlir/Conversion/MQTOptToQuantum/MQTOptToQuantum.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/MQTOptToQuantum/MQTOptToQuantum.h.inc"

} // namespace mlir::mqt::ir::conversions

#endif // LIB_CONVERSION_MQTOPTTOQUANTUM_MQTOPTTOQUANTUM_H_
