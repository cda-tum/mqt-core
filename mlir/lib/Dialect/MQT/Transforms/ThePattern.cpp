#include "mlir/Dialect/MQT/MQTOps.h"

#include <iostream>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>

/// Multi-step rewrite using "match" and "rewrite". This allows for separating
/// the concerns of matching and rewriting.
struct ThePattern : public mlir::OpRewritePattern<mlir::mqt::CustomOp> {
  explicit ThePattern(mlir::MLIRContext* context)
      : mlir::OpRewritePattern<mlir::mqt::CustomOp>(context) {}

  llvm::LogicalResult match(mlir::mqt::CustomOp op) const override {
    if (op.getGateName() == "Hadamard") {
      return llvm::success();
    }
    return llvm::failure();
  }

  void rewrite(mlir::mqt::CustomOp /*op*/,
               mlir::PatternRewriter& /*rewriter*/) const override {
    std::cout << "ATTENTION: Hadarmard detected!\n";
  }
};
