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
/// Analysis pattern that filters out all quantum operations from a given program.
struct QuantumFilterPattern final : OpRewritePattern<AllocOp> {
  std::set<Operation*>& handledOperations;

  explicit QuantumFilterPattern(MLIRContext* context, std::set<Operation*>& handled)
      : OpRewritePattern(context), handledOperations(handled) {}

  LogicalResult match(AllocOp op) const override {
    if (handledOperations.find(op) == handledOperations.end()) {
      return success();
    }
    
    return failure();
  }

  void rewrite(AllocOp op, PatternRewriter& rewriter) const override {
    std::cout << "ATTENTION: Hadarmard detected!\n";
    
    handledOperations.insert(op);
  }
};

void populateToQuantumComputationPatterns(RewritePatternSet& patterns, std::set<Operation*>& handled) {
  patterns.add<QuantumFilterPattern>(patterns.getContext(), handled);
}

} // namespace mlir::mqt
