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

struct QuantumSinkPushPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit QuantumSinkPushPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult match(UnitaryInterface op) const override {
    // Ensure that at least one user is a unitary operation.
    const auto& users = op->getUsers();
    auto* ownBlock = op->getBlock();
    if (std::none_of(users.begin(), users.end(), [&ownBlock](auto* user) {
          return mlir::isa<UnitaryInterface>(user) &&
                 user->getBlock() != ownBlock;
        })) {
      return mlir::failure();
    }
    return mlir::success();
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
               mlir::PatternRewriter& rewriter) const override {
    llvm::outs() << "looking at op: " << op << "\n";
    std::map<mlir::Block*, mlir::Operation*> blockToOp;
    for (auto* user : op->getUsers()) {
      blockToOp.insert(
          {user->getBlock(), user}); // Users per block are always <= 1.
    }

    auto* currentBlock = op->getBlock();
    auto* terminator = currentBlock->getTerminator();
    std::vector<mlir::Block*> nextBlocks = getBranchSuccessors(terminator);
    std::vector<mlir::Block*> newBlocks;

    for (auto* nextBlock : nextBlocks) {
      if (std::distance(nextBlock->getPredecessors().begin(),
                        nextBlock->getPredecessors().end()) == 1) {
        // We can simply shift the operation to the next block.
        mlir::Operation& blockStart = nextBlock->front();
        rewriter.setInsertionPoint(&blockStart);
        auto clone = rewriter.clone(*op);
        if (blockToOp.find(nextBlock) != blockToOp.end()) {
          replaceInputsWithClone(rewriter, op, clone, blockToOp[nextBlock]);
        } else {
          // TODO make accessible for later use
        }
        newBlocks.emplace_back(nextBlock);
      } else if (nextBlocks.size() > 1) {
        // We need to create a new block and insert the operation there to break
        // the critical edge.
        auto* newBlock = rewriter.createBlock(nextBlock->getPrevNode());
        rewriter.setInsertionPointToStart(newBlock);
        auto clone = rewriter.clone(*op);
        if (blockToOp.find(nextBlock) != blockToOp.end()) {
          replaceInputsWithClone(rewriter, op, clone, blockToOp[nextBlock]);
        }
        rewriter.create<mlir::cf::BranchOp>(op->getLoc(), nextBlock);
        newBlocks.emplace_back(newBlock);
      } else {
        // Operation has to stay here.
        newBlocks.emplace_back(nextBlock);
      }
    }

    // Update the original branch operation if necessary.
    if (newBlocks != nextBlocks) {
      auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(
          terminator); // must be a CondBranchOp.
      rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
          condBr, condBr.getCondition(), newBlocks[0], condBr.getTrueOperands(),
          newBlocks[1], condBr.getFalseOperands());
    }

    rewriter.eraseOp(op);
  }
};

/**
 * @brief Populates the given pattern set with the
 * `QuantumSinkPushPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateQuantumSinkPushPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<QuantumSinkPushPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
