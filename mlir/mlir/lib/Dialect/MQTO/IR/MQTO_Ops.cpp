#include "mlir/Dialect/MQTO/IR/MQTO_Dialect.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
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
  if (getInQubits().empty()) {
    return emitOpError() << "expected at least one input qubit";
  }
  if (getInQubits().size() == getOutQubits().size()) {
    return emitOpError() << "expected same number of input and output qubits";
  }
  return success();
}

LogicalResult AllocOp::verify() {
  if (!getNqubits() && !getNqubitsAttr().has_value()) {
    return emitOpError() << "expected a operand or attribute for nqubits";
  }
  if (getNqubits() && getNqubitsAttr().has_value()) {
    return emitOpError()
           << "expected either an operand or attribute for nqubits";
  }
  return success();
}

LogicalResult ExtractOp::verify() {
  if (!getIndex() && !getIndexAttr().has_value()) {
    return emitOpError() << "expected a operand or attribute for index";
  }
  if (getIndex() && getIndexAttr().has_value()) {
    return emitOpError() << "expected either an operand or attribute for index";
  }
  return success();
}

LogicalResult InsertOp::verify() {
  if (!getIndex() && !getIndexAttr().has_value()) {
    return emitOpError() << "expected a operand or attribute for index";
  }
  if (getIndex() && getIndexAttr().has_value()) {
    return emitOpError() << "expected either an operand or attribute for index";
  }
  return success();
}

} // namespace mlir::mqto
