#include "mlir/Dialect/MQT/IR/MQTOps.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"

#include <iostream>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir::mqt {

/// Multi-step rewrite using "match" and "rewrite". This allows for separating
/// the concerns of matching and rewriting.
struct ThePattern final : OpRewritePattern<CustomOp> {
  explicit ThePattern(MLIRContext* context) : OpRewritePattern(context) {}

  LogicalResult match(CustomOp op) const override {
    if (op.getGateName() == "Hadamard") {
      return success();
    }
    return failure();
  }

  void rewrite(CustomOp /*op*/, PatternRewriter& /*rewriter*/) const override {
    std::cout << "ATTENTION: Hadarmard detected!\n";
  }
};

void populateThePassPatterns(RewritePatternSet& patterns) {
  patterns.add<ThePattern>(patterns.getContext());
}

} // namespace mlir::mqt
