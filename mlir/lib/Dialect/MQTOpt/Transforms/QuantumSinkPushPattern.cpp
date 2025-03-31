/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <algorithm>
#include <iterator>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <vector>

namespace mqt::ir::opt {

/**
 * @brief This pattern is responsible to push Unitary operations through branch
 * operations into new blocks.
 *
 * This can be done in some circumstances, if a branch operation uses the result
 * of a unitary operation as a block argument.
 */
struct QuantumSinkPushPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit QuantumSinkPushPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult match(UnitaryInterface op) const override {
    // We only consider 1-qubit gates.
    if (op.getOutQubits().size() != 1) {
      return mlir::failure();
    }

    const auto& users = op->getUsers();
    if (users.empty()) {
      return mlir::failure();
    }

    auto* user = *users.begin();

    // This pattern only applies to operations in the same block.
    if (user->getBlock() != op->getBlock()) {
      return mlir::failure();
    }

    // The pattern can always be applied if the user is a conditional branch
    // operation.
    if (mlir::isa<mlir::cf::CondBranchOp>(user)) {
      return mlir::success();
    }

    // For normal branch operations, we can only apply the pattern if the
    // successor block only has one predecessor.
    if (auto branchOp = mlir::dyn_cast<mlir::cf::BranchOp>(user)) {
      auto* successor = branchOp.getDest();
      if (std::distance(successor->getPredecessors().begin(),
                        successor->getPredecessors().end()) == 1) {
        return mlir::success();
      }
    }
    return mlir::failure();
  }

  /**
   * @brief Pushes the given Unitary operation through a branch operation into
   * the given block.
   *
   * @param op The Unitary operation to push.
   * @param block The block to push the operation into.
   * @param blockArgs The arguments used to call the block.
   * @param rewriter The pattern rewriter to use.
   * @return The new arguments to call the block with.
   */
  std::vector<mlir::Value>
  pushIntoBlock(UnitaryInterface op, mlir::Block* block,
                mlir::OperandRange blockArgs,
                mlir::PatternRewriter& rewriter) const {
    std::vector<mlir::Value> newBlockArgs;
    // We start by inserting the operation at the beginning of the block.
    rewriter.setInsertionPointToStart(block);
    auto* clone = rewriter.clone(*op);

    // We iterate over all block args: If any of them involved the result of the
    // pushed operation, they can be removed, as the operation is now already in
    // the block. However, this means
    for (int i = static_cast<int>(blockArgs.size()) - 1; i >= 0; i--) {
      auto arg = blockArgs[i];
      auto found =
          std::find(op->getResults().begin(), op->getResults().end(), arg);
      if (found == op->getResults().end()) {
        newBlockArgs.emplace_back(arg);
        continue;
      }
      auto idx = std::distance(op->getResults().begin(), found);
      block->getArgument(i).replaceAllUsesWith(clone->getResult(idx));
      block->eraseArgument(i);
    }

    // Now, the block instead needs to be passed the inputs of the pushed
    // operation so we extend the block with new arguments.
    for (size_t i = 0; i < clone->getOperands().size(); i++) {
      auto in = clone->getOperand(i);
      auto newArg = block->addArgument(in.getType(), clone->getLoc());
      rewriter.modifyOpInPlace(clone, [&]() { clone->setOperand(i, newArg); });
      newBlockArgs.emplace_back(in);
    }

    return newBlockArgs;
  }

  /**
   * @brief Breaks a critical edge through the given branch operation by adding
   * a new block.
   *
   * @param oldTarget The target block to which the critical edge went.
   * @param branchOp The branch operation that caused the critical edge.
   * @param rewriter The pattern rewriter to use.
   * @return The new block that was created.
   */
  mlir::Block* breakCriticalEdge(mlir::Block* oldTarget,
                                 mlir::Operation* branchOp,
                                 mlir::PatternRewriter& rewriter) const {
    auto* newBlock = rewriter.createBlock(oldTarget->getParent());
    std::vector<mlir::Value> newBlockOutputs;
    for (auto arg : oldTarget->getArguments()) {
      auto newArg = newBlock->addArgument(arg.getType(), branchOp->getLoc());
      newBlockOutputs.emplace_back(newArg);
    }
    rewriter.setInsertionPointToEnd(newBlock);
    rewriter.create<mlir::cf::BranchOp>(branchOp->getLoc(), oldTarget,
                                        newBlockOutputs);
    return newBlock;
  }

  /**
   * @brief Pushes a Unitary operation through a single conditional branch
   * operation.
   *
   * @param op The Unitary operation to push.
   * @param condBranchOp The conditional branch operation to push through.
   * @param rewriter The pattern rewriter to use.
   */
  void rewriteCondBranch(UnitaryInterface op,
                         mlir::cf::CondBranchOp condBranchOp,
                         mlir::PatternRewriter& rewriter) const {
    auto* targetBlockTrue =
        std::distance(condBranchOp.getTrueDest()->getPredecessors().begin(),
                      condBranchOp.getTrueDest()->getPredecessors().end()) == 1
            ? condBranchOp.getTrueDest()
            : breakCriticalEdge(condBranchOp.getTrueDest(), condBranchOp,
                                rewriter);
    auto newBlockArgsTrue = pushIntoBlock(
        op, targetBlockTrue, condBranchOp.getTrueOperands(), rewriter);

    auto* targetBlockFalse =
        std::distance(condBranchOp.getFalseDest()->getPredecessors().begin(),
                      condBranchOp.getFalseDest()->getPredecessors().end()) == 1
            ? condBranchOp.getFalseDest()
            : breakCriticalEdge(condBranchOp.getFalseDest(), condBranchOp,
                                rewriter);
    auto newBlockArgsFalse = pushIntoBlock(
        op, targetBlockFalse, condBranchOp.getFalseOperands(), rewriter);

    rewriter.setInsertionPoint(condBranchOp);
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        condBranchOp, condBranchOp.getCondition(), targetBlockTrue,
        newBlockArgsTrue, targetBlockFalse, newBlockArgsFalse);
    rewriter.eraseOp(op);
  }

  /**
   * @brief Pushes a Unitary operation through a single branch operation.
   *
   * The successor of this branch will always have exactly one predecessor as
   * per the match function. Therefore, we do not have to worry about critical
   * edges.
   *
   * @param op The Unitary operation to push.
   * @param branchOp The branch operation to push through.
   * @param rewriter The pattern rewriter to use.
   */
  void rewriteBranch(UnitaryInterface op, mlir::cf::BranchOp branchOp,
                     mlir::PatternRewriter& rewriter) const {
    auto newBlockArgs =
        pushIntoBlock(op, branchOp.getDest(), branchOp.getOperands(), rewriter);
    rewriter.modifyOpInPlace(branchOp, [&]() {
      branchOp.getDestOperandsMutable().assign(newBlockArgs);
    });
    rewriter.eraseOp(op);
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {
    auto* user = *op->getUsers().begin();
    if (auto condBranchOp = mlir::dyn_cast<mlir::cf::CondBranchOp>(user)) {
      rewriteCondBranch(op, condBranchOp, rewriter);
      return;
    } else {
      rewriteBranch(op, mlir::dyn_cast<mlir::cf::BranchOp>(user), rewriter);
      return;
    }
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
