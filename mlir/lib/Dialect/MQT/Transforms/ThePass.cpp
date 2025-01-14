#include "ThePass.h.inc"
#include "mlir/Dialect/MQT/MQTOps.h"

#include <mlir/IR/PatternMatch.h>

static mlir::Value
performThePass(mlir::PatternRewriter& rewriter, mlir::Operation* op,
               mlir::Value params, mlir::Value in_qubits,
               mlir::Attribute gate_name, mlir::Attribute adjoint,
               mlir::Value in_ctrl_qubits, mlir::Value in_ctrl_values) {
  return rewriter.create<mlir::mqt::CustomOp>(
      op->getLoc(), {params, in_qubits, in_ctrl_qubits, in_ctrl_values},
      {in_qubits, gate_name});
}
