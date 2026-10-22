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
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Support/Passes.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <string>

using namespace mlir;

namespace {
struct ResetCase {
  const char* name;
  const char* body;
  size_t resets;
};

class RemoveRedundantResetsTest : public testing::TestWithParam<ResetCase> {
protected:
  MLIRContext context;

  RemoveRedundantResetsTest() {
    context.loadDialect<arith::ArithDialect, cf::ControlFlowDialect,
                        func::FuncDialect, scf::SCFDialect, qco::QCODialect,
                        qtensor::QTensorDialect>();
  }

  OwningOpRef<ModuleOp> parse(StringRef body) {
    return parseSourceString<ModuleOp>((R"mlir(module {
          func.func private @source() -> !qco.qubit
          func.func @test(%i: index) -> !qco.qubit {
            %q = func.call @source() : () -> !qco.qubit
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c4 = arith.constant 4 : index
        )mlir" + body + "}}")
                                           .str(),
                                       &context);
  }

  static size_t countResets(ModuleOp moduleOp) {
    size_t count = 0;
    moduleOp.walk([&](qco::ResetOp) { ++count; });
    return count;
  }
};

TEST_P(RemoveRedundantResetsTest, RemovesOnlyKnownZeroResets) {
  auto moduleOp = parse(GetParam().body);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  PassManager manager(&context);
  manager.addPass(qco::createRemoveRedundantResets());
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  EXPECT_EQ(countResets(*moduleOp), GetParam().resets);
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  EXPECT_EQ(countResets(*moduleOp), GetParam().resets);
}

INSTANTIATE_TEST_SUITE_P(ForwardAnalysis, RemoveRedundantResetsTest,
                         testing::Values(ResetCase{"ScalarAllocation", R"mlir(
          %a = qco.alloc : !qco.qubit
          %r = qco.reset %a : !qco.qubit
          qco.sink %r : !qco.qubit
          return %q : !qco.qubit
        )mlir",
                                                   0},
                                         ResetCase{"FreshTensor", R"mlir(
          %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
          %t1, %a = qtensor.extract %t[%c0] : tensor<4x!qco.qubit>
          %r = qco.reset %a : !qco.qubit
          qtensor.dealloc %t1 : tensor<4x!qco.qubit>
          qco.sink %r : !qco.qubit
          return %q : !qco.qubit
        )mlir",
                                                   0},
                                         ResetCase{"OtherSlotTouched", R"mlir(
          %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
          %t1 = qtensor.insert %q into %t[%c0] : tensor<4x!qco.qubit>
          %t2, %a = qtensor.extract %t1[%c1] : tensor<4x!qco.qubit>
          %r = qco.reset %a : !qco.qubit
          qtensor.dealloc %t2 : tensor<4x!qco.qubit>
          return %r : !qco.qubit
        )mlir",
                                                   0},
                                         ResetCase{"ComputedIndex", R"mlir(
          %index = arith.addi %c0, %c1 : index
          %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
          %t1 = qtensor.insert %q into %t[%c0] : tensor<4x!qco.qubit>
          %t2, %a = qtensor.extract %t1[%index] : tensor<4x!qco.qubit>
          %r = qco.reset %a : !qco.qubit
          qtensor.dealloc %t2 : tensor<4x!qco.qubit>
          return %r : !qco.qubit
        )mlir",
                                                   0},
                                         ResetCase{"SameSlotWritten", R"mlir(
          %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
          %t1 = qtensor.insert %q into %t[%c0] : tensor<4x!qco.qubit>
          %t2, %a = qtensor.extract %t1[%c0] : tensor<4x!qco.qubit>
          %r = qco.reset %a : !qco.qubit
          qtensor.dealloc %t2 : tensor<4x!qco.qubit>
          return %r : !qco.qubit
        )mlir",
                                                   1},
                                         ResetCase{"SameSlotRead", R"mlir(
          %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
          %t1, %a = qtensor.extract %t[%c0] : tensor<4x!qco.qubit>
          %t2, %b = qtensor.extract %t1[%c0] : tensor<4x!qco.qubit>
          %r = qco.reset %b : !qco.qubit
          qco.sink %a : !qco.qubit
          qco.sink %r : !qco.qubit
          qtensor.dealloc %t2 : tensor<4x!qco.qubit>
          return %q : !qco.qubit
        )mlir",
                                                   1},
                                         ResetCase{"UnknownRead", R"mlir(
          %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
          %t1, %a = qtensor.extract %t[%i] : tensor<4x!qco.qubit>
          %t2, %b = qtensor.extract %t1[%c0] : tensor<4x!qco.qubit>
          %ra = qco.reset %a : !qco.qubit
          %rb = qco.reset %b : !qco.qubit
          qco.sink %ra : !qco.qubit
          qco.sink %rb : !qco.qubit
          qtensor.dealloc %t2 : tensor<4x!qco.qubit>
          return %q : !qco.qubit
        )mlir",
                                                   2},
                                         ResetCase{"UnknownWrite", R"mlir(
          %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
          %t1 = qtensor.insert %q into %t[%i] : tensor<4x!qco.qubit>
          %t2, %a = qtensor.extract %t1[%c0] : tensor<4x!qco.qubit>
          %r = qco.reset %a : !qco.qubit
          qtensor.dealloc %t2 : tensor<4x!qco.qubit>
          return %r : !qco.qubit
        )mlir",
                                                   1},
                                         ResetCase{"ResetResult", R"mlir(
          %r1 = qco.reset %q : !qco.qubit
          %r2 = qco.reset %r1 : !qco.qubit
          %r3 = qco.reset %r2 : !qco.qubit
          return %r3 : !qco.qubit
        )mlir",
                                                   1},
                                         ResetCase{"GateResult", R"mlir(
          %a = qco.alloc : !qco.qubit
          %h = qco.h %a : !qco.qubit
          %r = qco.reset %h : !qco.qubit
          qco.sink %r : !qco.qubit
          return %q : !qco.qubit
        )mlir",
                                                   1},
                                         ResetCase{"BlockArgument", R"mlir(
          %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
          cf.br ^next(%t : tensor<4x!qco.qubit>)
        ^next(%next: tensor<4x!qco.qubit>):
          %t1, %a = qtensor.extract %next[%c0] : tensor<4x!qco.qubit>
          %r = qco.reset %a : !qco.qubit
          qco.sink %r : !qco.qubit
          qtensor.dealloc %t1 : tensor<4x!qco.qubit>
          return %q : !qco.qubit
        )mlir",
                                                   1},
                                         ResetCase{"UnknownProducer", R"mlir(
          %t = qtensor.from_elements %q : tensor<1x!qco.qubit>
          %t1, %a = qtensor.extract %t[%c0] : tensor<1x!qco.qubit>
          %r = qco.reset %a : !qco.qubit
          qtensor.dealloc %t1 : tensor<1x!qco.qubit>
          return %r : !qco.qubit
        )mlir",
                                                   1},
                                         ResetCase{"NestedAllocation", R"mlir(
          scf.execute_region {
            %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
            %t1, %a = qtensor.extract %t[%c0] : tensor<4x!qco.qubit>
            %r = qco.reset %a : !qco.qubit
            qco.sink %r : !qco.qubit
            qtensor.dealloc %t1 : tensor<4x!qco.qubit>
            scf.yield
          }
          return %q : !qco.qubit
        )mlir",
                                                   0}),
                         [](const testing::TestParamInfo<ResetCase>& info) {
                           return info.param.name;
                         });

TEST_F(RemoveRedundantResetsTest, CleanupExposesComputedConstantIndices) {
  auto moduleOp = parse(R"mlir(
    %index = arith.addi %c0, %c1 : index
    %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
    %t1 = qtensor.insert %q into %t[%c0] : tensor<4x!qco.qubit>
    %t2, %a = qtensor.extract %t1[%index] : tensor<4x!qco.qubit>
    %r = qco.reset %a : !qco.qubit
    qtensor.dealloc %t2 : tensor<4x!qco.qubit>
    return %r : !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(*moduleOp)));
  EXPECT_EQ(countResets(*moduleOp), 0U);
  EXPECT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
}

TEST_F(RemoveRedundantResetsTest, CleanupFoldsRoundTripsBeforeResetAnalysis) {
  auto moduleOp = parse(R"mlir(
    %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
    %t1, %a = qtensor.extract %t[%c0] : tensor<4x!qco.qubit>
    %t2 = qtensor.insert %a into %t1[%c0] : tensor<4x!qco.qubit>
    %t3, %b = qtensor.extract %t2[%c0] : tensor<4x!qco.qubit>
    %r = qco.reset %b : !qco.qubit
    qtensor.dealloc %t3 : tensor<4x!qco.qubit>
    qco.sink %r : !qco.qubit
    return %q : !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(*moduleOp)));
  EXPECT_EQ(countResets(*moduleOp), 0U);
  EXPECT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
}

TEST_F(RemoveRedundantResetsTest, RegistersTextualPasses) {
  auto moduleOp = parse(R"mlir(
    %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
    %t1, %a = qtensor.extract %t[%c0] : tensor<4x!qco.qubit>
    %r = qco.reset %a : !qco.qubit
    qtensor.dealloc %t1 : tensor<4x!qco.qubit>
    qco.sink %r : !qco.qubit
    return %q : !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(
      runPassPipeline(*moduleOp, "remove-redundant-resets,qtensor-normalize")));
  EXPECT_EQ(countResets(*moduleOp), 0U);
  EXPECT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
}
TEST_F(RemoveRedundantResetsTest, LeavesGraphRegionsUnchanged) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      %q = qco.alloc : !qco.qubit
      %r = qco.reset %q : !qco.qubit
      qco.sink %r : !qco.qubit
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  auto alloc = *moduleOp->getOps<qco::AllocOp>().begin();
  auto reset = *moduleOp->getOps<qco::ResetOp>().begin();
  reset->moveBefore(alloc);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  PassManager pm(&context);
  pm.addPass(qco::createRemoveRedundantResets());
  ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
  EXPECT_EQ(countResets(*moduleOp), 1U);
  EXPECT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
}
} // namespace
