#include <algorithm>
#include <map>
#include <vector>

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace mqt::ir::opt {

struct BranchOperationPushdownPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit BranchOperationPushdownPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult match(UnitaryInterface op) const override {
    // Ensure that at least one user is a unitary operation.
    const auto& users = op->getUsers();
    if (std::none_of(users.begin(), users.end(), [](auto* user) {
      return mlir::isa<UnitaryInterface>(user);
    })) {
      return mlir::failure();
    }

    // Ensure that the original operation is in the unique predecessor block of each user.
    auto* ownBlock = op->getBlock();
    for(auto* user : users) {
      auto* userBlock = user->getBlock();
      if (std::distance(userBlock->getPredecessors().begin(), userBlock->getPredecessors().end()) != 1) {
        return mlir::failure();
      }
      if (*userBlock->getPredecessors().begin() != ownBlock) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  void replaceInputsWithClone(mlir::PatternRewriter& rewriter,
                              mlir::Operation* original,
                              mlir::Operation* clone,
                              mlir::Operation* user) const {
    for (size_t i = 0; i < user->getOperands().size(); i++) {
      const auto& operand = user->getOperand(i);
      const auto found = std::find(original->getResults().begin(),
                                   original->getResults().end(), operand);
      if (found == original->getResults().end()) {
        continue;
      }
      const auto idx = std::distance(original->getResults().begin(), found);
      rewriter.modifyOpInPlace(user, [&] {
        user->setOperand(i, clone->getResults()[idx]);
      });
    }

  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {
    llvm::outs() << "looking at op: " << op << "\n";
    std::map<mlir::Block*, mlir::Operation*> blockToOp;
    for(auto* user : op->getUsers()) {
      blockToOp.insert({user->getBlock(), user}); // Users per block are always <= 1.
    }

    auto* terminator = op->getBlock()->getTerminator();

    std::vector<mlir::Block*> nextBlocks;
    if(auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
      nextBlocks.emplace_back(condBr.getTrueDest());
      nextBlocks.emplace_back(condBr.getFalseDest());
    } else if(auto br = mlir::dyn_cast<mlir::cf::BranchOp>(terminator)) {
      nextBlocks.emplace_back(br.getDest());
    }

    for(auto* nextBlock : nextBlocks) {
      mlir::Operation& blockStart = nextBlock->front();
      rewriter.setInsertionPoint(&blockStart);
      auto clone = rewriter.clone(*op);
      if(blockToOp.find(nextBlock) != blockToOp.end()) {
        replaceInputsWithClone(rewriter, op, clone, blockToOp[nextBlock]);
      }
    }
    rewriter.eraseOp(op);
  }
};

/**
 * @brief Populates the given pattern set with the
 * `BranchOperationPushdownPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateBranchOperationPushdownPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<BranchOperationPushdownPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
