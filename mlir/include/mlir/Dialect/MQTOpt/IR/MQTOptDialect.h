#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTOpt/IR/MQTOptInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOps.h.inc"
