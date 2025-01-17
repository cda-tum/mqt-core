#include "ir/QuantumComputation.hpp"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/IR/Operation.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <set>
#include <utility>

namespace mlir::mqt {

#define GEN_PASS_DEF_TOQUANTUMCOMPUTATION
#include "mlir/Dialect/MQT/Transforms/Passes.h.inc"

struct ToQuantumComputation final
    : impl::ToQuantumComputationBase<ToQuantumComputation> {

  std::set<Operation*> handledOperations;
  qc::QuantumComputation circuit;

  void runOnOperation() override {
    // Get the current operation being operated on.
    Operation* op = getOperation();
    MLIRContext* ctx = &getContext();

    // Define the set of patterns to use.
    RewritePatternSet patterns(ctx);
    populateToQuantumComputationPatterns(patterns, handledOperations, circuit);
    populateFromQuantumComputationPatterns(patterns, handledOperations,
                                           circuit);

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::mqt
