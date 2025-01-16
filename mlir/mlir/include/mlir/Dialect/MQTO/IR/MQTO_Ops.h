#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTO/IR/MQTO_OpsTypes.h.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/MQTO/IR/MQTO_Ops.h.inc"
