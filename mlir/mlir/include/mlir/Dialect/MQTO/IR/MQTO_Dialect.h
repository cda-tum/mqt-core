#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

//===----------------------------------------------------------------------===//
// Dialect declarations
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTO/IR/MQTO_OpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTO/IR/MQTO_OpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Enum declarations
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTO/IR/MQTO_OpsEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MQTO/IR/MQTO_OpsAttributes.h.inc"

//===----------------------------------------------------------------------===//
// Operation declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTO/IR/MQTO_Ops.h.inc"
