#include "mlir/Dialect/MQT/IR/MQTOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include <iostream>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>


namespace mlir::mqt {

/// Pattern to transform `quantum.custom` multi-qubit gate to MQT equivalent.
struct AllocRewritePattern : public OpRewritePattern<AllocOp> {
  explicit AllocRewritePattern(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult match(AllocOp op) const override {
    // NOTE: This pattern is currently only checking for the Hadamard gate.
    return success();
  }

  void rewrite(AllocOp op, PatternRewriter &rewriter) const override {
    std::cout << "APPLYING: AllocRewritePattern\n";

    // Extract the size operand or attribute from the original operation
    auto sizeOperand = op.getNqubits(); // Optional operand
    auto sizeAttr = op.getNqubitsAttrAttr(); // Optional attribute

    // Get the result type (QubitRegisterType)
    auto resultType = ::mqt::ir::opt::QubitRegisterType::get(rewriter.getContext());

    // Create a new AllocOp using the appropriate builder
    ::mqt::ir::opt::AllocOp mqtoPlaceholder;
    if (sizeOperand) {
        // Use the builder with size as an operand
        mqtoPlaceholder = rewriter.create<::mqt::ir::opt::AllocOp>(
            op.getLoc(), resultType, sizeOperand, nullptr);
    } else if (sizeAttr) {
        // Use the builder with size as an attribute
        mqtoPlaceholder = rewriter.create<::mqt::ir::opt::AllocOp>(
            op.getLoc(), resultType, nullptr, sizeAttr);
    } else {
        // Emit a failure if neither operand nor attribute is provided
        rewriter.notifyMatchFailure(op, "AllocOp must have either a size operand or a size attribute");
        return;
    }

    // Replace the original operation with the new AllocOp
    rewriter.replaceOp(op, mqtoPlaceholder);
  }
};

/// Populate patterns for the transformation pass.
void populatePassWithAllocRewritePattern(RewritePatternSet& patterns) {
  patterns.add<AllocRewritePattern>(patterns.getContext());
}

} // namespace mlir::mqt
