#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/IR/Operation.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace mlir::mqt {

#define GEN_PASS_DEF_QUANTUM_TO_MQTOPT
#include "mlir/Dialect/MQT/Transforms/Passes.h.inc"

struct QuantumToMQTOpt final : impl::QuantumToMQTOptBase<QuantumToMQTOpt> {

    void getDependentDialects(DialectRegistry& registry) const override {
    // Register the required dialects.
    registry.insert<::mqt::ir::opt::MQTOptDialect>();
  }

  void runOnOperation() override {
    // Get the current operation being operated on.
    Operation* op = getOperation();
    MLIRContext* ctx = &getContext();

    // Define the set of patterns to use.
    RewritePatternSet thePatterns(ctx);
    ctx->getOrLoadDialect<::mqt::ir::opt::MQTOptDialect>();
    populatePassWithAllocRewritePattern(thePatterns);

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsAndFoldGreedily(op, std::move(thePatterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::mqt
