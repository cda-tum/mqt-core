#include "mlir/Dialect/MQT/IR/MQTOps.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"

#include <iostream>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <set>

namespace mlir::mqt {
/// Multi-step rewrite using "match" and "rewrite". This allows for separating
/// the concerns of matching and rewriting.
struct ThePattern final : OpRewritePattern<CustomOp> {
  DenseSet<Operation*>& handled;

  explicit ThePattern(MLIRContext* context,
                      DenseSet<Operation*>& handledOperations)
      : OpRewritePattern(context), handled(handledOperations) {}

  LogicalResult match(CustomOp op) const override {
    if (handled.contains(op)) {
      return failure(); // Skip already handled operations
    }
    if (op.getGateName() == "Hadamard") {
      return success();
    }
    return failure();
  }

  void rewrite(CustomOp op, PatternRewriter& /*rewriter*/) const override {
    handled.insert(op);
    std::cout << "ATTENTION: Hadarmard detected!\n";
  }
};

void populateThePassPatterns(RewritePatternSet& patterns,
                             DenseSet<Operation*>& handledOperations) {
  patterns.add<ThePattern>(patterns.getContext(), handledOperations);
}

} // namespace mlir::mqt
