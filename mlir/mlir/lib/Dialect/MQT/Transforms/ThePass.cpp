#include "mlir/Dialect/MQT/Transforms/Passes.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace mlir {

#define GEN_PASS_DEF_THEPASS
#include "mlir/Dialect/MQT/Transforms/Passes.h.inc"

namespace mqt {

struct ThePass : PassWrapper<ThePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Get the current operation being operated on.
    const ModuleOp op = getOperation();
    MLIRContext* ctx = &getContext();

    // Define the set of patterns to use.
    RewritePatternSet thePatterns(ctx);
    populateThePassPatterns(thePatterns);

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsAndFoldGreedily(op, std::move(thePatterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt

} // namespace mlir
