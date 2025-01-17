#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

//===----------------------------------------------------------------------===//
// Dialect declarations
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Enum declarations
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsAttributes.h.inc"

//===----------------------------------------------------------------------===//
// Operation declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOps.h.inc"
