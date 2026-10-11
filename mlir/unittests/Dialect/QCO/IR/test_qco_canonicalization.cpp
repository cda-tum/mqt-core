/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"

#include "ExactUnitaryTest.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <complex>
#include <cstddef>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {

class QCOCanonicalizationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  }

  OwningOpRef<ModuleOp> singleQubitFunction() {
    return parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @main(%q: !qco.qubit) -> !qco.qubit {
          return %q : !qco.qubit
        }
      }
    )mlir",
                                       &context_);
  }

  OwningOpRef<ModuleOp> twoQubitFunction() {
    return parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @main(%q0: !qco.qubit, %q1: !qco.qubit)
            -> (!qco.qubit, !qco.qubit) {
          return %q0, %q1 : !qco.qubit, !qco.qubit
        }
      }
    )mlir",
                                       &context_);
  }

  void canonicalize(ModuleOp moduleOp) {
    ASSERT_TRUE(succeeded(verify(moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp)));
    PassManager manager(&context_);
    manager.addPass(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(manager.run(moduleOp)));
    ASSERT_TRUE(succeeded(verify(moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp)));
  }

  template <typename GateOp> void checkPairCancellation() {
    auto moduleOp = singleQubitFunction();
    ASSERT_TRUE(moduleOp);
    auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
    auto returned = cast<func::ReturnOp>(function.getBody().front().back());
    OpBuilder builder(returned);
    auto first =
        GateOp::create(builder, function.getLoc(), function.getArgument(0));
    auto second = GateOp::create(builder, function.getLoc(), first.getResult());
    returned->setOperand(0, second.getResult());
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));

    RewritePatternSet patterns(&context_);
    GateOp::getCanonicalizationPatterns(patterns, &context_);
    FrozenRewritePatternSet frozen(std::move(patterns));
    PatternApplicator applicator(frozen);
    applicator.applyDefaultCostModel();
    PatternRewriter rewriter(&context_);
    rewriter.setInsertionPoint(first);
    ASSERT_TRUE(succeeded(applicator.matchAndRewrite(first, rewriter)));
    // Pair cancellation must preserve linearity before any later DCE runs.
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));
    EXPECT_TRUE(function.getBody().getOps<GateOp>().empty());
    EXPECT_EQ(returned.getOperand(0), function.getArgument(0));
  }

  template <typename GateOp, typename InverseOp = GateOp>
  void checkControlledPairCancellation() {
    auto moduleOp = twoQubitFunction();
    ASSERT_TRUE(moduleOp);
    auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
    auto returned = cast<func::ReturnOp>(function.getBody().front().back());
    OpBuilder builder(returned);
    auto first = CtrlOp::create(
        builder, function.getLoc(), function.getArgument(0),
        function.getArgument(1), [&](Value target) -> Value {
          return GateOp::create(builder, function.getLoc(), target).getResult();
        });
    auto second = CtrlOp::create(
        builder, function.getLoc(), first.getControlsOut().front(),
        first.getTargetsOut().front(), [&](Value target) -> Value {
          return InverseOp::create(builder, function.getLoc(), target)
              .getResult();
        });
    returned->setOperands(second.getResults());
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));
    OwningOpRef<ModuleOp> original(moduleOp->clone());

    RewritePatternSet patterns(&context_);
    CtrlOp::getCanonicalizationPatterns(patterns, &context_);
    FrozenRewritePatternSet frozen(std::move(patterns));
    PatternApplicator applicator(frozen);
    applicator.applyDefaultCostModel();
    PatternRewriter rewriter(&context_);
    rewriter.setInsertionPoint(first);
    ASSERT_TRUE(succeeded(applicator.matchAndRewrite(first, rewriter)));
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));
    EXPECT_TRUE(function.getBody().getOps<CtrlOp>().empty());
    EXPECT_TRUE(llvm::equal(returned.getOperands(), function.getArguments()));
    ::mqt::test::expectFullUnitaryEqual(*original, *moduleOp, 2);
  }
};

TEST_F(QCOCanonicalizationTest, HadamardPairCancellationPreservesLinearity) {
  checkPairCancellation<HOp>();
}
TEST_F(QCOCanonicalizationTest, PauliXPairCancellationPreservesLinearity) {
  checkPairCancellation<XOp>();
}
TEST_F(QCOCanonicalizationTest, PauliYPairCancellationPreservesLinearity) {
  checkPairCancellation<YOp>();
}
TEST_F(QCOCanonicalizationTest, PauliZPairCancellationPreservesLinearity) {
  checkPairCancellation<ZOp>();
}

TEST_F(QCOCanonicalizationTest, ControlledInversePairsPreserveLinearity) {
  checkControlledPairCancellation<HOp>();
  checkControlledPairCancellation<XOp>();
  checkControlledPairCancellation<YOp>();
  checkControlledPairCancellation<ZOp>();
  checkControlledPairCancellation<SOp, SdgOp>();
  checkControlledPairCancellation<SdgOp, SOp>();
  checkControlledPairCancellation<TOp, TdgOp>();
  checkControlledPairCancellation<TdgOp, TOp>();
  checkControlledPairCancellation<SXOp, SXdgOp>();
  checkControlledPairCancellation<SXdgOp, SXOp>();
}

TEST_F(QCOCanonicalizationTest, ControlledPairsRespectWiresAndBoundaries) {
  for (const auto* boundary : {
           "target gate",
           "control gate",
           "barrier",
           "reversed wires",
           "different control",
           "different gate",
           "relative phase",
       }) {
    SCOPED_TRACE(boundary);
    auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @main(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit)
            -> (!qco.qubit, !qco.qubit, !qco.qubit) {
          return %q0, %q1, %q2 : !qco.qubit, !qco.qubit, !qco.qubit
        }
      }
    )mlir",
                                                &context_);
    ASSERT_TRUE(moduleOp);
    auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
    auto returned = cast<func::ReturnOp>(function.getBody().front().back());
    OpBuilder builder(returned);
    const auto loc = function.getLoc();
    auto first =
        CtrlOp::create(builder, loc, function.getArgument(0),
                       function.getArgument(1), [&](Value target) -> Value {
                         return XOp::create(builder, loc, target).getResult();
                       });
    SmallVector<Value> qubits(first.getResults());
    qubits.push_back(function.getArgument(2));
    const StringRef kind(boundary);
    if (kind == "target gate") {
      qubits[1] = HOp::create(builder, loc, qubits[1]).getResult();
    } else if (kind == "control gate") {
      qubits[0] = HOp::create(builder, loc, qubits[0]).getResult();
    } else if (kind == "barrier") {
      auto barrier = BarrierOp::create(builder, loc, ValueRange{qubits[0]});
      qubits[0] = barrier.getResult(0);
    }
    const size_t control = kind == "reversed wires"      ? 1
                           : kind == "different control" ? 2
                                                         : 0;
    const size_t target = kind == "reversed wires" ? 0 : 1;
    auto second =
        CtrlOp::create(builder, loc, qubits[control], qubits[target],
                       [&](Value input) -> Value {
                         if (kind == "relative phase") {
                           auto theta = arith::ConstantOp::create(
                               builder, loc, builder.getF64FloatAttr(0.125));
                           GPhaseOp::create(builder, loc, theta.getResult());
                         }
                         if (kind == "different gate") {
                           return YOp::create(builder, loc, input).getResult();
                         }
                         return XOp::create(builder, loc, input).getResult();
                       });
    qubits[control] = second.getControlsOut().front();
    qubits[target] = second.getTargetsOut().front();
    returned->setOperands(qubits);
    OwningOpRef<ModuleOp> original(moduleOp->clone());

    ASSERT_NO_FATAL_FAILURE(canonicalize(*moduleOp));
    EXPECT_EQ(llvm::range_size(function.getBody().getOps<CtrlOp>()), 2);
    ::mqt::test::expectFullUnitaryEqual(*original, *moduleOp, 3);
  }
}

TEST_F(QCOCanonicalizationTest, NestedControlledPairsCancelExactly) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %a0, %a1, %a2 = qco.ctrl(%q0) targets(%a = %q1, %b = %q2) {
          %c, %d = qco.ctrl(%a) targets(%t = %b) {
            %x = qco.x %t : !qco.qubit
            qco.yield %x : !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit})
          qco.yield %c, %d : !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit})
        %b0, %b1, %b2 = qco.ctrl(%a0) targets(%a = %a1, %b = %a2) {
          %c, %d = qco.ctrl(%a) targets(%t = %b) {
            %x = qco.x %t : !qco.qubit
            qco.yield %x : !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit})
          qco.yield %c, %d : !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit})
        return %b0, %b1, %b2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  OwningOpRef<ModuleOp> original(moduleOp->clone());
  ASSERT_NO_FATAL_FAILURE(canonicalize(*moduleOp));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  EXPECT_TRUE(function.getBody().getOps<CtrlOp>().empty());
  auto returned = cast<func::ReturnOp>(function.getBody().front().back());
  EXPECT_TRUE(llvm::equal(returned.getOperands(), function.getArguments()));
  ::mqt::test::expectFullUnitaryEqual(*original, *moduleOp, 3);
}

TEST_F(QCOCanonicalizationTest, PairCancellationPreservesDifferentGates) {
  auto moduleOp = singleQubitFunction();
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  auto returned = cast<func::ReturnOp>(function.getBody().front().back());
  OpBuilder builder(returned);
  auto first = HOp::create(builder, function.getLoc(), function.getArgument(0));
  auto second = XOp::create(builder, function.getLoc(), first.getResult());
  returned->setOperand(0, second.getResult());

  ASSERT_NO_FATAL_FAILURE(canonicalize(*moduleOp));
  EXPECT_EQ(returned.getOperand(0), second.getResult());
  EXPECT_EQ(second.getQubitIn(), first.getResult());
}

TEST_F(QCOCanonicalizationTest, IdentityFoldsToItsInput) {
  auto moduleOp = singleQubitFunction();
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  auto returned = cast<func::ReturnOp>(function.getBody().front().back());
  OpBuilder builder(returned);
  auto identity =
      IdOp::create(builder, function.getLoc(), function.getArgument(0));
  returned->setOperand(0, identity.getResult());
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));

  SmallVector<OpFoldResult> results;
  ASSERT_TRUE(succeeded(identity->fold({Attribute{}}, results)));
  ASSERT_EQ(results.size(), 1U);
  EXPECT_EQ(cast<Value>(results.front()), function.getArgument(0));
  ASSERT_NO_FATAL_FAILURE(canonicalize(*moduleOp));
  EXPECT_EQ(returned.getOperand(0), function.getArgument(0));
}

TEST_F(QCOCanonicalizationTest, DenseIdentityFoldPreservesOperandOrder) {
  auto moduleOp = twoQubitFunction();
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  auto returned = cast<func::ReturnOp>(function.getBody().front().back());
  OpBuilder builder(returned);
  const auto matrixType =
      RankedTensorType::get({4, 4}, ComplexType::get(builder.getF64Type()));
  std::array<std::complex<double>, 16> values{};
  for (size_t index = 0; index < 4; ++index) {
    values[(4 * index) + index] = 1.0;
  }
  const auto matrix = DenseElementsAttr::get(
      matrixType, ArrayRef<std::complex<double>>(values));
  auto identity = UnitaryOp::create(
      builder, function.getLoc(),
      ValueRange{function.getArgument(1), function.getArgument(0)}, matrix);
  returned->setOperands(
      ValueRange{identity.getOutputQubit(1), identity.getOutputQubit(0)});
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));

  SmallVector<OpFoldResult> results;
  ASSERT_TRUE(succeeded(identity->fold({Attribute{}, Attribute{}}, results)));
  ASSERT_EQ(results.size(), 2U);
  EXPECT_EQ(cast<Value>(results[0]), function.getArgument(1));
  EXPECT_EQ(cast<Value>(results[1]), function.getArgument(0));
  ASSERT_NO_FATAL_FAILURE(canonicalize(*moduleOp));
  EXPECT_EQ(returned.getOperand(0), function.getArgument(0));
  EXPECT_EQ(returned.getOperand(1), function.getArgument(1));
}

TEST_F(QCOCanonicalizationTest, DenseIdentityFoldPreservesNonzeroPhase) {
  const std::array<std::complex<double>, 3> phases{
      std::complex<double>{-1.0, 0.0},
      std::polar(1.0, 1e-11),
      std::polar(1.0, 1e-16),
  };
  for (const auto phase : phases) {
    SCOPED_TRACE(testing::Message() << "phase factor " << phase);
    auto original = singleQubitFunction();
    ASSERT_TRUE(original);
    auto function = original->lookupSymbol<func::FuncOp>("main");
    auto returned = cast<func::ReturnOp>(function.getBody().front().back());
    OpBuilder builder(returned);
    const auto matrixType =
        RankedTensorType::get({2, 2}, ComplexType::get(builder.getF64Type()));
    const std::array<std::complex<double>, 4> values{phase, 0.0, 0.0, phase};
    const auto matrix = DenseElementsAttr::get(
        matrixType, ArrayRef<std::complex<double>>(values));
    auto phaseOp =
        UnitaryOp::create(builder, function.getLoc(),
                          ValueRange{function.getArgument(0)}, matrix);
    returned->setOperand(0, phaseOp.getOutputQubit(0));
    ASSERT_TRUE(succeeded(verify(*original)));
    ASSERT_TRUE(succeeded(verifyLinearity(*original)));

    SmallVector<OpFoldResult> results;
    EXPECT_TRUE(failed(phaseOp->fold({Attribute{}}, results)));
    EXPECT_TRUE(results.empty());
    OwningOpRef<ModuleOp> rewritten(original->clone());
    ASSERT_NO_FATAL_FAILURE(canonicalize(*rewritten));
    auto rewrittenFunction = rewritten->lookupSymbol<func::FuncOp>("main");
    auto rewrittenReturn =
        cast<func::ReturnOp>(rewrittenFunction.getBody().front().back());
    auto rewrittenPhase =
        rewrittenReturn.getOperand(0).getDefiningOp<UnitaryOp>();
    ASSERT_TRUE(rewrittenPhase);
    EXPECT_EQ(rewrittenPhase.getMatrix(), matrix);
    ::mqt::test::expectFullUnitaryEqual(*original, *rewritten, 1);
  }
}

} // namespace
