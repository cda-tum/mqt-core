#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include <algorithm>
#include <map>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <vector>

namespace mqt::ir::opt {

/**
 * @brief This pattern is responsible to shift Unitary operations into the block their results are used in.
*/
struct QuantumSinkShiftPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit QuantumSinkShiftPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult match(UnitaryInterface op) const override {
    // We only consider 1-qubit gates.
    if(op.getOutQubits().size() != 1) {
      return mlir::failure();
    }
    
    // Ensure that there is at least one user.
    const auto& users = op->getUsers();
    if(users.empty()) {
      return mlir::failure();
    }

    // There will only ever be one user
    auto* user = *users.begin();
    return user->getBlock() != op->getBlock() ? mlir::success()
                                              : mlir::failure();
  }

  void replaceInputsWithClone(mlir::PatternRewriter& rewriter,
                              mlir::Operation* original, mlir::Operation* clone,
                              mlir::Operation* user) const {
    for (size_t i = 0; i < user->getOperands().size(); i++) {
      const auto& operand = user->getOperand(i);
      const auto found = std::find(original->getResults().begin(),
                                   original->getResults().end(), operand);
      if (found == original->getResults().end()) {
        continue;
      }
      const auto idx = std::distance(original->getResults().begin(), found);
      rewriter.modifyOpInPlace(
          user, [&] { user->setOperand(i, clone->getResults()[idx]); });
    }
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {
    llvm::outs() << "PUSH: looking at op: " << op << "\n";
    
    auto* user = *op->getUsers().begin();
    auto* block = user->getBlock();

    rewriter.setInsertionPoint(&block->front());
    auto* clone = rewriter.clone(*op);
    //replaceInputsWithClone(rewriter, op, clone, user);
    op->replaceAllUsesWith(clone);
    rewriter.eraseOp(op);
  }
};

/**
 * @brief Populates the given pattern set with the
 * `QuantumSinkShiftPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateQuantumSinkShiftPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<QuantumSinkShiftPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
