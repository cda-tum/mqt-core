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
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Dialect/QTensor/Transforms/Passes.h"
#include "mqt/Support/Passes.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <iterator>
#include <string>

using namespace mlir;

namespace {
class QTensorNormalizationTest : public testing::Test {
protected:
  MLIRContext context;

  void SetUp() override {
    context.loadDialect<arith::ArithDialect, cf::ControlFlowDialect,
                        func::FuncDialect, scf::SCFDialect, qco::QCODialect,
                        qtensor::QTensorDialect>();
  }

  OwningOpRef<ModuleOp> parse(StringRef body) {
    return parseSourceString<ModuleOp>((R"mlir(module {
          func.func @test(%i: index)
              -> (tensor<4x!qco.qubit>, !qco.qubit, !qco.qubit) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c4 = arith.constant 4 : index
            %t = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
            %a = qco.alloc : !qco.qubit
            %b = qco.alloc : !qco.qubit
        )mlir" + body + "}}")
                                           .str(),
                                       &context);
  }

  void normalize(ModuleOp moduleOp) {
    ASSERT_TRUE(succeeded(verify(moduleOp)));
    ASSERT_TRUE(succeeded(qco::verifyLinearity(moduleOp)));
    PassManager manager(&context);
    manager.addPass(qtensor::createNormalizeQTensorPass());
    ASSERT_TRUE(succeeded(manager.run(moduleOp)));
    EXPECT_TRUE(succeeded(verify(moduleOp)));
    EXPECT_TRUE(succeeded(qco::verifyLinearity(moduleOp)));
  }

  template <typename Op> static size_t count(ModuleOp moduleOp) {
    size_t result = 0;
    moduleOp.walk([&](Op) { ++result; });
    return result;
  }

  static std::string print(ModuleOp moduleOp) {
    std::string result;
    llvm::raw_string_ostream stream(result);
    moduleOp.print(stream);
    return result;
  }
};

TEST_F(QTensorNormalizationTest, ForwardsScalarsAcrossOtherSlots) {
  auto moduleOp = parse(R"mlir(
    %t1 = qtensor.insert %a into %t[%c0] : tensor<4x!qco.qubit>
    %t2, %x = qtensor.extract %t1[%c1] : tensor<4x!qco.qubit>
    %t3 = qtensor.insert %b into %t2[%c2] : tensor<4x!qco.qubit>
    %t4, %y = qtensor.extract %t3[%c0] : tensor<4x!qco.qubit>
    return %t4, %x, %y : tensor<4x!qco.qubit>, !qco.qubit, !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_NO_FATAL_FAILURE(normalize(*moduleOp));
  EXPECT_EQ(count<qtensor::ExtractOp>(*moduleOp), 1U);
  EXPECT_EQ(count<qtensor::InsertOp>(*moduleOp), 1U);
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto ret = cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(ret.getOperand(2),
            (*function.getOps<qco::AllocOp>().begin()).getResult());
  auto snapshot = print(*moduleOp);
  ASSERT_NO_FATAL_FAILURE(normalize(*moduleOp));
  EXPECT_EQ(print(*moduleOp), snapshot);
}

TEST_F(QTensorNormalizationTest, FoldsLocalRoundTripBeforeReordering) {
  auto moduleOp = parse(R"mlir(
    %index = arith.addi %c0, %c1 : index
    %t1, %idle = qtensor.extract %t[%c1] : tensor<4x!qco.qubit>
    %t2 = qtensor.insert %idle into %t1[%index] : tensor<4x!qco.qubit>
    %t3 = qtensor.insert %a into %t2[%c0] : tensor<4x!qco.qubit>
    %t4, %x = qtensor.extract %t3[%c0] : tensor<4x!qco.qubit>
    return %t4, %x, %b : tensor<4x!qco.qubit>, !qco.qubit, !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_NO_FATAL_FAILURE(normalize(*moduleOp));
  EXPECT_EQ(count<qtensor::ExtractOp>(*moduleOp), 0U);
  EXPECT_EQ(count<qtensor::InsertOp>(*moduleOp), 0U);
}

TEST_F(QTensorNormalizationTest, FoldsRoundTripExposedByForwarding) {
  auto moduleOp = parse(R"mlir(
    %t1, %q = qtensor.extract %t[%c0] : tensor<4x!qco.qubit>
    %t2 = qtensor.insert %a into %t1[%c1] : tensor<4x!qco.qubit>
    %t3, %x = qtensor.extract %t2[%c1] : tensor<4x!qco.qubit>
    %t4 = qtensor.insert %q into %t3[%c0] : tensor<4x!qco.qubit>
    return %t4, %x, %b : tensor<4x!qco.qubit>, !qco.qubit, !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_NO_FATAL_FAILURE(normalize(*moduleOp));
  EXPECT_EQ(count<qtensor::ExtractOp>(*moduleOp), 0U);
  EXPECT_EQ(count<qtensor::InsertOp>(*moduleOp), 0U);
}

TEST_F(QTensorNormalizationTest,
       CleanupNormalizesAccessesExposedByLoopInlining) {
  auto moduleOp = parse(R"mlir(
    %t1, %x = qtensor.extract %t[%c0] : tensor<4x!qco.qubit>
    %t2, %y = qtensor.extract %t1[%c1] : tensor<4x!qco.qubit>
    %t3 = qtensor.insert %x into %t2[%c0] : tensor<4x!qco.qubit>
    %t4 = qtensor.insert %y into %t3[%c1] : tensor<4x!qco.qubit>
    %loop = scf.for %iv = %c0 to %c1 step %c1
        iter_args(%iter = %t4) -> (tensor<4x!qco.qubit>) {
      %s1, %q0 = qtensor.extract %iter[%c0] : tensor<4x!qco.qubit>
      %h = qco.h %q0 : !qco.qubit
      %s2 = qtensor.insert %h into %s1[%c0] : tensor<4x!qco.qubit>
      %s3, %q1 = qtensor.extract %s2[%c1] : tensor<4x!qco.qubit>
      %v = qco.x %q1 : !qco.qubit
      %s4 = qtensor.insert %v into %s3[%c1] : tensor<4x!qco.qubit>
      scf.yield %s4 : tensor<4x!qco.qubit>
    }
    return %loop, %a, %b : tensor<4x!qco.qubit>, !qco.qubit, !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(*moduleOp)));
  EXPECT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  EXPECT_EQ(count<scf::ForOp>(*moduleOp), 0U);
  EXPECT_EQ(count<qtensor::ExtractOp>(*moduleOp), 2U);
  EXPECT_EQ(count<qtensor::InsertOp>(*moduleOp), 2U);
}

TEST_F(QTensorNormalizationTest, EvaluatesComputedIndicesWithoutRewritingThem) {
  auto moduleOp = parse(R"mlir(
    %index = arith.subi %c1, %c1 : index
    %t1 = qtensor.insert %a into %t[%c0] : tensor<4x!qco.qubit>
    %t2, %x = qtensor.extract %t1[%index] : tensor<4x!qco.qubit>
    return %t2, %x, %b : tensor<4x!qco.qubit>, !qco.qubit, !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_NO_FATAL_FAILURE(normalize(*moduleOp));
  EXPECT_EQ(count<qtensor::ExtractOp>(*moduleOp), 0U);
  EXPECT_EQ(count<qtensor::InsertOp>(*moduleOp), 0U);
  EXPECT_EQ(count<arith::SubIOp>(*moduleOp), 1U);
}

TEST_F(QTensorNormalizationTest, RestartsAfterUnknownIndex) {
  auto moduleOp = parse(R"mlir(
    %t1 = qtensor.insert %a into %t[%c0] : tensor<4x!qco.qubit>
    %t2, %x = qtensor.extract %t1[%i] : tensor<4x!qco.qubit>
    %t3 = qtensor.insert %b into %t2[%c2] : tensor<4x!qco.qubit>
    %t4, %y = qtensor.extract %t3[%c2] : tensor<4x!qco.qubit>
    return %t4, %x, %y : tensor<4x!qco.qubit>, !qco.qubit, !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_NO_FATAL_FAILURE(normalize(*moduleOp));
  EXPECT_EQ(count<qtensor::ExtractOp>(*moduleOp), 1U);
  EXPECT_EQ(count<qtensor::InsertOp>(*moduleOp), 1U);
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto ret = cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(ret.getOperand(2),
            (*std::next(function.getOps<qco::AllocOp>().begin())).getResult());
}

TEST_F(QTensorNormalizationTest, PreservesRepeatedWritesAndMetadata) {
  auto moduleOp = parse(R"mlir(
    %t1 = qtensor.insert %a into %t[%c0] {test.tag = "first"}
        : tensor<4x!qco.qubit> loc("first-write")
    %t2 = qtensor.insert %b into %t1[%c0] : tensor<4x!qco.qubit>
    %t3, %x = qtensor.extract %t2[%c1] {test.tag = "read"}
        : tensor<4x!qco.qubit> loc("read")
    %t4, %y = qtensor.extract %t3[%c0] : tensor<4x!qco.qubit>
    return %t4, %x, %y : tensor<4x!qco.qubit>, !qco.qubit, !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_NO_FATAL_FAILURE(normalize(*moduleOp));
  EXPECT_EQ(count<qtensor::InsertOp>(*moduleOp), 1U);
  EXPECT_EQ(count<qtensor::ExtractOp>(*moduleOp), 1U);
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto insert = *function.getBody().front().getOps<qtensor::InsertOp>().begin();
  EXPECT_EQ(insert.getScalar(),
            (*function.getOps<qco::AllocOp>().begin()).getResult());
  EXPECT_EQ(insert->getAttrOfType<StringAttr>("test.tag").getValue(), "first");
  EXPECT_EQ(insert.getLoc(),
            NameLoc::get(StringAttr::get(&context, "first-write")));
  auto extract =
      *function.getBody().front().getOps<qtensor::ExtractOp>().begin();
  EXPECT_EQ(extract->getAttrOfType<StringAttr>("test.tag").getValue(), "read");
  EXPECT_EQ(extract.getLoc(), NameLoc::get(StringAttr::get(&context, "read")));
  auto snapshot = print(*moduleOp);
  ASSERT_NO_FATAL_FAILURE(normalize(*moduleOp));
  EXPECT_EQ(print(*moduleOp), snapshot);
}

TEST_F(QTensorNormalizationTest, ReadsRepeatedWritesInReverseOrder) {
  auto moduleOp = parse(R"mlir(
    %t1 = qtensor.insert %a into %t[%c0] : tensor<4x!qco.qubit>
    %t2 = qtensor.insert %b into %t1[%c0] : tensor<4x!qco.qubit>
    %t3, %x = qtensor.extract %t2[%c0] : tensor<4x!qco.qubit>
    %t4, %y = qtensor.extract %t3[%c0] : tensor<4x!qco.qubit>
    return %t4, %x, %y : tensor<4x!qco.qubit>, !qco.qubit, !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_NO_FATAL_FAILURE(normalize(*moduleOp));
  EXPECT_EQ(count<qtensor::ExtractOp>(*moduleOp), 0U);
  EXPECT_EQ(count<qtensor::InsertOp>(*moduleOp), 0U);
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto ret = cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(ret.getOperand(0),
            (*function.getOps<qtensor::AllocOp>().begin()).getResult());
  EXPECT_EQ(ret.getOperand(1),
            (*std::next(function.getOps<qco::AllocOp>().begin())).getResult());
  EXPECT_EQ(ret.getOperand(2),
            (*function.getOps<qco::AllocOp>().begin()).getResult());
}

TEST_F(QTensorNormalizationTest, LeavesBatchedAccessesUnchanged) {
  auto moduleOp = parse(R"mlir(
    %t1, %x = qtensor.extract %t[%c0] : tensor<4x!qco.qubit>
    %t2, %y = qtensor.extract %t1[%c1] : tensor<4x!qco.qubit>
    %t3 = qtensor.insert %a into %t2[%c0] : tensor<4x!qco.qubit>
    %t4 = qtensor.insert %b into %t3[%c1] : tensor<4x!qco.qubit>
    return %t4, %x, %y : tensor<4x!qco.qubit>, !qco.qubit, !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto snapshot = print(*moduleOp);
  ASSERT_NO_FATAL_FAILURE(normalize(*moduleOp));
  EXPECT_EQ(print(*moduleOp), snapshot);
}

TEST_F(QTensorNormalizationTest, StopsAtBlockBoundary) {
  auto moduleOp = parse(R"mlir(
    %t1 = qtensor.insert %a into %t[%c0] : tensor<4x!qco.qubit>
    cf.br ^next(%t1 : tensor<4x!qco.qubit>)
  ^next(%next: tensor<4x!qco.qubit>):
    %t2, %x = qtensor.extract %next[%c0] : tensor<4x!qco.qubit>
    return %t2, %x, %b : tensor<4x!qco.qubit>, !qco.qubit, !qco.qubit
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto snapshot = print(*moduleOp);
  ASSERT_NO_FATAL_FAILURE(normalize(*moduleOp));
  EXPECT_EQ(print(*moduleOp), snapshot);
}

TEST_F(QTensorNormalizationTest, LeavesGraphRegionsUnchanged) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tensor = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
      %r0, %q0 = qtensor.extract %tensor[%c0] : tensor<2x!qco.qubit>
      %h = qco.h %q0 : !qco.qubit
      %t0 = qtensor.insert %h into %r0[%c0] : tensor<2x!qco.qubit>
      %r1, %q1 = qtensor.extract %t0[%c1] : tensor<2x!qco.qubit>
      qco.sink %q1 : !qco.qubit
      qtensor.dealloc %r1 : tensor<2x!qco.qubit>
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  auto insert = *moduleOp->getOps<qtensor::InsertOp>().begin();
  auto extract = cast<qtensor::ExtractOp>(*insert.getResult().user_begin());
  extract->moveBefore(insert);
  normalize(*moduleOp);
  EXPECT_EQ(extract.getTensor(), insert.getResult());
  EXPECT_TRUE(extract->isBeforeInBlock(insert));
}
} // namespace
