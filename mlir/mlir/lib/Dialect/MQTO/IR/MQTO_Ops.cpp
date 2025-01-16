#include "mlir/Dialect/MQTO/IR/MQTO_Ops.h"

#include "mlir/Support/LLVM.h"

#include <mlir/IR/Builders.h>

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
