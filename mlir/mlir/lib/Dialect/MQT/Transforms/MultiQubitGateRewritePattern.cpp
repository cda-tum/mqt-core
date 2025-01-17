#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/IR/MQTOps.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include <iostream>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir::mqt {

/// Pattern to transform `quantum.custom` multi-qubit gate to MQT equivalent.
struct MultiQubitGateRewritePattern : public OpRewritePattern<CustomOp> {
  explicit MultiQubitGateRewritePattern(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult match(CustomOp op) const override {
    // NOTE: This pattern is currently only checking for the CNOT gate.
    return op.getGateName() == "CNOT" ? success() : failure();
  }

  void rewrite(CustomOp op, PatternRewriter& rewriter) const override {
    std::cout << "APPLYING: MultiQubitGateRewritePattern\n";

    // Extract the input qubit for the operation.
    auto qubits = op.getInQubits();

    // Replace with a new MQT operation.
    auto mqtoPlaceholder = rewriter.create<CustomOp>(
        op.getLoc(), "MQTOMulitQubitGatePlaceholder", mlir::ValueRange{qubits});

    // Replace the original operation with the new MQT operation.
    rewriter.replaceOp(op, mqtoPlaceholder);
  }
};

/// Populate patterns for the transformation pass.
void populatePassWithMultiQubitGateRewritePattern(RewritePatternSet& patterns) {
  patterns.add<MultiQubitGateRewritePattern>(patterns.getContext());
}

} // namespace mlir::mqt
