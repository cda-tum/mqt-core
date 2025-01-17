#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LLVM.h>

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsDialect.cpp.inc"

void mqt::ir::opt::MQTOptDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MQTOpt/IR/MQTOptOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

namespace mqt::ir::opt {

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

} // namespace mqt::ir::opt
