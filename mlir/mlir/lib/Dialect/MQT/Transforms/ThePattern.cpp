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
    if (op.getGateName() == "PauliZ") {
      return success();
    }
    return failure();
  }

  void rewrite(CustomOp op, PatternRewriter& rewriter) const override {
    handled.insert(op);
    std::cout << "ATTENTION: Hadarmard detected!\n";
    
    // Replace with a new operation
    auto qubits = op.getInQubits();
    CustomOp hadamard_0 = rewriter.create<CustomOp>(op.getLoc(), TypeRange{QubitType}, mlir::ValueRange{"Hadamard", qubits});

    qubits = hadamard_0.getOutQubits();
    CustomOp pauli_x = rewriter.create<CustomOp>(op.getLoc(), TypeRange{mlir::StringRef, mlir::ValueRange}, mlir::ValueRange{"PauliX", qubits});

    qubits = pauli_x.getOutQubits();
    CustomOp hadamard_1 = rewriter.create<CustomOp>(op.getLoc(), TypeRange{mlir::StringRef, mlir::ValueRange}, mlir::ValueRange{"Hadamard", qubits});

    rewriter.replaceOp<CustomOp>(op, mlir::ValueRange{hadamard_0});
  }
};

void populateThePassPatterns(RewritePatternSet& patterns,
                             DenseSet<Operation*>& handledOperations) {
  patterns.add<ThePattern>(patterns.getContext(), handledOperations);
}

} // namespace mlir::mqt
