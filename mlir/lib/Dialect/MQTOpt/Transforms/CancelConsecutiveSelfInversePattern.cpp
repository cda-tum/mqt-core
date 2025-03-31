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

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace mqt::ir::opt {

/**
 * @brief This pattern attempty to cancel consecutive self-inverse operations.
 */
struct CancelConsecutiveSelfInversePattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit CancelConsecutiveSelfInversePattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult match(UnitaryInterface op) const override {
    if (!(mlir::isa<XOp>(op) || mlir::isa<YOp>(op) || mlir::isa<ZOp>(op) ||
          mlir::isa<HOp>(op) || mlir::isa<IOp>(op) || mlir::isa<SWAPOp>(op))) {
      return mlir::failure();
    }
    const auto& users = op->getUsers();
    if (std::distance(users.begin(), users.end()) != 1) {
      return mlir::failure();
    }
    auto user = *users.begin();
    if (op->getName() != user->getName()) {
      return mlir::failure();
    }
    auto unitaryUser = mlir::dyn_cast<UnitaryInterface>(user);
    if (op.getOutQubits() != unitaryUser.getInQubits()) {
      return mlir::failure();
    }
    return mlir::success();
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {
    auto user = mlir::dyn_cast<UnitaryInterface>(
        *op->getUsers().begin()); // We always have exactly one user.
    const auto& childUsers = user->getUsers();

    for (const auto& childUser : childUsers) {
      for (size_t i = 0; i < childUser->getOperands().size(); i++) {
        const auto& operand = childUser->getOperand(i);
        const auto found = std::find(user.getOutQubits().begin(),
                                     user.getOutQubits().end(), operand);
        if (found == user.getOutQubits().end()) {
          continue;
        }
        const auto idx = std::distance(user.getOutQubits().begin(), found);
        rewriter.modifyOpInPlace(childUser, [&] {
          childUser->setOperand(i, op.getInQubits()[idx]);
        });
      }
    }

    rewriter.eraseOp(user);
    rewriter.eraseOp(op);
  }
};

/**
 * @brief Populates the given pattern set with the
 * `CancelConsecutiveSelfInversePattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateCancelSelfInversePatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<CancelConsecutiveSelfInversePattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
