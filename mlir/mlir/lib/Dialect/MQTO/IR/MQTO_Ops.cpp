#include "mlir/Dialect/MQTO/IR/MQTO_Ops.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/Support/LLVM.h>

//===----------------------------------------------------------------------===//
// MQT dialect definitions.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTO/IR/MQTO_OpsDialect.cpp.inc"

void mlir::mqto::MQTODialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/MQTO/IR/MQTO_OpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MQTO/IR/MQTO_Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// MQT type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTO/IR/MQTO_OpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTO/IR/MQTO_Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

namespace mlir::mqto {

LogicalResult OperationOp::verify() {
  return success(getInQubits().size() > 0 &&
                 (getInQubits().size() == getOutQubits().size()));
}

LogicalResult AllocOp::verify() {
  return success((getNumOperands() == 1) ^ getNqubitsAttr().has_value());
}

LogicalResult ExtractOp::verify() {
  return success((getNumOperands() == 2) ^ getIndexAttr().has_value());
}

LogicalResult InsertOp::verify() {
  return success((getNumOperands() == 3) ^ getIndexAttr().has_value());
}

} // namespace mlir::mqto
