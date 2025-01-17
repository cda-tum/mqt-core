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
struct SingleQubitGateRewritePattern : public OpRewritePattern<CustomOp> {
  explicit SingleQubitGateRewritePattern(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult match(CustomOp op) const override {
    // NOTE: This pattern is currently only checking for the Hadamard gate.
    return op.getGateName() == "Hadamard" ? success() : failure();
  }

  void rewrite(CustomOp op, PatternRewriter& rewriter) const override {
    std::cout << "APPLYING: SingleQubitGateRewritePattern\n";

    // Extract the input qubit for the operation.
    auto qubit = op.getInQubits()[0];

    // Replace with a new MQT operation.
    auto mqtoPlaceholder = rewriter.create<CustomOp>(
        op.getLoc(), "MQTOSingleQubitGatePlaceholder", mlir::ValueRange{qubit});

    // Replace the original operation with the new MQT operation.
    rewriter.replaceOp(op, mqtoPlaceholder);
  }
};

/// Populate patterns for the transformation pass.
void populatePassWithSingleQubitGateRewritePattern(
    RewritePatternSet& patterns) {
  patterns.add<SingleQubitGateRewritePattern>(patterns.getContext());
}

} // namespace mlir::mqt
