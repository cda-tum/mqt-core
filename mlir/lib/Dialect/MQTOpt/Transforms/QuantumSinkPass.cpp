#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/Operation.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_QUANTUMSINKPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

/**
 * @brief This pass attempty to sink quantum operations into the block where their results are used.
 */
struct QuantumSinkPass final : impl::QuantumSinkPassBase<QuantumSinkPass> {

  void runOnOperation() override {
    // Get the current operation being operated on.
    mlir::Operation* op = getOperation();
    mlir::MLIRContext* ctx = &getContext();

    // Define the set of patterns to use.
    mlir::RewritePatternSet patterns(ctx);
    populateQuantumSinkShiftPatterns(patterns);
    populateQuantumSinkPushPatterns(patterns);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(
            mlir::applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
