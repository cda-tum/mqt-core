#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/IR/Operation.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace mlir::mqt {

#define GEN_PASS_DEF_THEPASS
#include "mlir/Dialect/MQT/Transforms/Passes.h.inc"

struct ThePass final : impl::ThePassBase<ThePass> {

  DenseSet<Operation*> handled;

  void runOnOperation() override {
    // Get the current operation being operated on.
    Operation* op = getOperation();
    MLIRContext* ctx = &getContext();

    // Define the set of patterns to use.
    RewritePatternSet thePatterns(ctx);
    populateThePassPatterns(thePatterns, handled);

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsAndFoldGreedily(op, std::move(thePatterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::mqt
