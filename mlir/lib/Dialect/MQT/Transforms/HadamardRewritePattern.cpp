#include "mlir/Dialect/MQT/IR/MQTOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include <iostream>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <set>

namespace mlir::mqt {

/// Pattern to transform `quantum.custom` Hadamard to MQT equivalent.
struct HadamardRewritePattern : public OpRewritePattern<CustomOp> {
  explicit HadamardRewritePattern(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult match(CustomOp op) const override {
    // Match only "Hadamard" gate in the quantum dialect.
    return op.getGateName() == "Hadamard" ? success() : failure();
  }

  void rewrite(CustomOp op, PatternRewriter& rewriter) const override {
    std::cout << "APPLYING: HadamardRewritePattern\n";

    // Extract the input qubit for the operation.
    auto qubit = op.getInQubits()[0];

    // Replace with a new MQT operation.
    auto mqtoHadamard = rewriter.create<CustomOp>(op.getLoc(), "MQTOHadamard", mlir::ValueRange{qubit});
    
    // Replace the original operation with the new MQT operation.
    rewriter.replaceOp(op, mqtoHadamard);
  }
};

/// Populate patterns for the transformation pass.
void populateThePassWithHadamardRewritePattern(RewritePatternSet& patterns) {
  patterns.add<HadamardRewritePattern>(patterns.getContext());
}

} // namespace mlir::mqt
