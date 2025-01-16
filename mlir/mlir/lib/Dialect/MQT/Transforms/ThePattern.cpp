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
/// Multi-step rewrite using "match" and "rewrite". This allows for separating
/// the concerns of matching and rewriting.
struct ThePattern final : OpRewritePattern<CustomOp> {
  explicit ThePattern(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult match(CustomOp op) const override {
    if (op.getGateName() == "PauliZ") {
      return success();
    }
    return failure();
  }

  void rewrite(CustomOp op, PatternRewriter& rewriter) const override {
    std::cout << "ATTENTION: Hadarmard detected!\n";
    
    // Replace with a new operation
    auto qubits = op.getInQubits()[0];

    auto hadamard_0 = rewriter.create<CustomOp>(op.getLoc(), "Hadamard", mlir::ValueRange{qubits});
    
    qubits = hadamard_0.getOutQubits()[0];
    auto pauli_x = rewriter.create<CustomOp>(op.getLoc(), "PauliX", mlir::ValueRange{qubits});

    qubits = pauli_x.getOutQubits()[0];
    auto hadamard_1 = rewriter.create<CustomOp>(op.getLoc(), "Hadamard", mlir::ValueRange{qubits});

    rewriter.replaceOp(op, hadamard_1);
  }
};

void populateThePassPatterns(RewritePatternSet& patterns) {
  patterns.add<ThePattern>(patterns.getContext());
}

} // namespace mlir::mqt
