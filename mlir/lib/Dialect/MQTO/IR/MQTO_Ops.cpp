#include "mlir/Dialect/MQTO/IR/MQTO_Dialect.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LLVM.h>

//===----------------------------------------------------------------------===//
// MQT dialect definitions.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTO/IR/MQTO_OpsDialect.cpp.inc"

void mqtmlir::mqto::MQTODialect::initialize() {
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

namespace mqtmlir::mqto {

mlir::LogicalResult OperationOp::verify() {
  if (getInQubits().empty()) {
    return emitOpError() << "expected at least one input qubit";
  }
  if (getInQubits().size() != getOutQubits().size()) {
    return emitOpError() << "expected same number of input and output qubits";
  }
  return mlir::success();
}

mlir::LogicalResult AllocOp::verify() {
  if (!getSize() && !getSizeAttr().has_value()) {
    return emitOpError() << "expected an operand or attribute for size";
  }
  if (getSize() && getSizeAttr().has_value()) {
    return emitOpError() << "expected either an operand or attribute for size";
  }
  return mlir::success();
}

mlir::LogicalResult ExtractOp::verify() {
  if (!getIndex() && !getIndexAttr().has_value()) {
    return emitOpError() << "expected an operand or attribute for index";
  }
  if (getIndex() && getIndexAttr().has_value()) {
    return emitOpError() << "expected either an operand or attribute for index";
  }
  return mlir::success();
}

mlir::LogicalResult InsertOp::verify() {
  if (!getIndex() && !getIndexAttr().has_value()) {
    return emitOpError() << "expected an operand or attribute for index";
  }
  if (getIndex() && getIndexAttr().has_value()) {
    return emitOpError() << "expected either an operand or attribute for index";
  }
  return mlir::success();
}

} // namespace mqtmlir::mqto
