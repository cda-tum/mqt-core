#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include <algorithm>
#include <map>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <vector>

namespace mqt::ir::opt {

struct QuantumSinkMergePattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit QuantumSinkMergePattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult match(UnitaryInterface op) const override {
    return mlir::failure();
  }

  std::vector<mlir::Block*> getBranchSuccessors(mlir::Operation* op) const {
    std::vector<mlir::Block*> blocks;
    if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(op)) {
      blocks.emplace_back(condBr.getTrueDest());
      blocks.emplace_back(condBr.getFalseDest());
    } else if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(op)) {
      blocks.emplace_back(br.getDest());
    }
    return blocks;
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {}
};

/**
 * @brief Populates the given pattern set with the
 * `QuantumSinkMergePattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateQuantumSinkMergePatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<QuantumSinkMergePattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
