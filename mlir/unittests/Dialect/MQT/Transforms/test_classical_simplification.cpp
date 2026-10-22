/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file
/// Tests for block-local memory and scoped condition simplification.

#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/APInt.h"

using namespace mlir;

namespace {

class ClassicalSimplificationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<cbit::CBitDialect, arith::ArithDialect,
                         func::FuncDialect, qco::QCODialect, scf::SCFDialect>();
  }

  LogicalResult canonicalize(ModuleOp moduleOp) {
    PassManager manager(&context_);
    manager.addPass(mlir::mqt::createSimplifyClassicalControl());
    manager.addPass(createCanonicalizerPass());
    return manager.run(moduleOp);
  }
};

TEST_F(ClassicalSimplificationTest, ForwardsStoredBitAcrossReadOnlySnapshot) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test() -> (i1, i2) {
        %c0 = arith.constant 0 : index
        %true = arith.constant true
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        cbit.store %true, %reg[%c0] : !cbit.reg<2>
        %snapshot = cbit.read %reg : !cbit.reg<2> -> i2
        %bit = cbit.load %reg[%c0] : !cbit.reg<2>
        return %bit, %snapshot : i1, i2
      }
    }
  )mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  APInt value;
  ASSERT_TRUE(matchPattern(returned.getOperand(0), m_ConstantInt(&value)));
  EXPECT_TRUE(value.isOne());
  EXPECT_TRUE(returned.getOperand(1).getDefiningOp<cbit::ReadOp>());
}

TEST_F(ClassicalSimplificationTest, WholeRegisterWriteReplacesStoredBits) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%value: i65) -> (i1, i65) {
        %c0 = arith.constant 0 : index
        %true = arith.constant true
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<65>
        cbit.store %true, %reg[%c0] : !cbit.reg<65>
        cbit.write %value, %reg : i65, !cbit.reg<65>
        %snapshot = cbit.read %reg : !cbit.reg<65> -> i65
        %bit = cbit.load %reg[%c0] : !cbit.reg<65>
        return %bit, %snapshot : i1, i65
      }
    }
  )mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_TRUE(returned.getOperand(0).getDefiningOp<cbit::LoadOp>());
  EXPECT_EQ(returned.getOperand(1), function.getArgument(0));
}

TEST_F(ClassicalSimplificationTest,
       RepeatedLoadsPreserveRegistersIndicesAndInterveningStores) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%reg: !cbit.reg<2>, %other: !cbit.reg<2>,
                      %index: index, %other_index: index, %value: i1)
          -> (i1, i1, i1, i1, i1) {
        %first = cbit.load %reg[%index] : !cbit.reg<2>
        %repeated = cbit.load %reg[%index] : !cbit.reg<2>
        %other_reg = cbit.load %other[%index] : !cbit.reg<2>
        %other_bit = cbit.load %reg[%other_index] : !cbit.reg<2>
        cbit.store %value, %reg[%index] : !cbit.reg<2>
        %written = cbit.load %reg[%index] : !cbit.reg<2>
        return %first, %repeated, %other_reg, %other_bit, %written
            : i1, i1, i1, i1, i1
      }
    }
  )mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), returned.getOperand(1));
  auto first = returned.getOperand(0).getDefiningOp<cbit::LoadOp>();
  auto otherReg = returned.getOperand(2).getDefiningOp<cbit::LoadOp>();
  auto otherBit = returned.getOperand(3).getDefiningOp<cbit::LoadOp>();
  ASSERT_TRUE(first);
  ASSERT_TRUE(otherReg);
  ASSERT_TRUE(otherBit);
  EXPECT_EQ(first.getReg(), function.getArgument(0));
  EXPECT_EQ(first.getIndex(), function.getArgument(2));
  EXPECT_EQ(otherReg.getReg(), function.getArgument(1));
  EXPECT_EQ(otherReg.getIndex(), function.getArgument(2));
  EXPECT_EQ(otherBit.getReg(), function.getArgument(0));
  EXPECT_EQ(otherBit.getIndex(), function.getArgument(3));
  EXPECT_EQ(returned.getOperand(4), function.getArgument(4));
}

TEST_F(ClassicalSimplificationTest, RepeatedStoresRetainOtherKnownBits) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @test(%reg: !cbit.reg<2>, %a: i1, %b: i1) -> (i1, i1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cbit.store %a, %reg[%c0] : !cbit.reg<2>
      cbit.store %a, %reg[%c1] : !cbit.reg<2>
      cbit.store %b, %reg[%c0] : !cbit.reg<2>
      %first = cbit.load %reg[%c0] : !cbit.reg<2>
      %second = cbit.load %reg[%c1] : !cbit.reg<2>
      return %first, %second : i1, i1
    }
  })mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned = cast<func::ReturnOp>(function.front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(2));
  EXPECT_EQ(returned.getOperand(1), function.getArgument(1));
}

TEST_F(ClassicalSimplificationTest, ConditionsRemainScopedToTheirBranches) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%condition: i1, %other: i1) -> (i1, i1) {
        %result = qco.if %condition args() -> i1 {
          %nested = qco.if %condition args() -> i1 {
            qco.yield %other : i1
          } else args() {
            qco.yield %condition : i1
          }
          qco.yield %nested : i1
        } else args() {
          qco.yield %condition : i1
        }
        return %result, %condition : i1, i1
      }
    }
  )mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned = cast<func::ReturnOp>(function.front().getTerminator());
  auto branch = returned.getOperand(0).getDefiningOp<qco::IfOp>();
  ASSERT_TRUE(branch);
  EXPECT_EQ(branch.getCondition(), function.getArgument(0));
  EXPECT_EQ(branch.thenYield().getOperand(0), function.getArgument(1));
  APInt falseValue;
  ASSERT_TRUE(matchPattern(branch.elseYield().getOperand(0),
                           m_ConstantInt(&falseValue)));
  EXPECT_TRUE(falseValue.isZero());
  EXPECT_EQ(returned.getOperand(1), function.getArgument(0));
}

TEST_F(ClassicalSimplificationTest, ConditionsPropagateThroughOtherRegions) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @test(%condition: i1) -> i1 {
      %result = qco.if %condition args() -> i1 {
        %nested = scf.execute_region -> i1 {
          scf.yield %condition : i1
        }
        qco.yield %nested : i1
      } else args() {
        qco.yield %condition : i1
      }
      return %result : i1
    }
  })mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  PassManager manager(&context_);
  manager.addPass(mqt::createSimplifyClassicalControl());
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned = cast<func::ReturnOp>(function.front().getTerminator());
  auto branch = returned.getOperand(0).getDefiningOp<qco::IfOp>();
  ASSERT_TRUE(branch);
  auto nested =
      branch.thenYield().getOperand(0).getDefiningOp<scf::ExecuteRegionOp>();
  ASSERT_TRUE(nested);
  APInt value;
  ASSERT_TRUE(
      matchPattern(nested.getRegion().front().getTerminator()->getOperand(0),
                   m_ConstantInt(&value)));
  EXPECT_TRUE(value.isOne());
  ASSERT_TRUE(
      matchPattern(branch.elseYield().getOperand(0), m_ConstantInt(&value)));
  EXPECT_TRUE(value.isZero());
  EXPECT_EQ(branch.getCondition(), function.getArgument(0));
}

TEST_F(ClassicalSimplificationTest, ForwardsAcrossEffectFreeRegions) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @test(%reg: !cbit.reg<2>, %seed: i1, %condition: i1) -> (i1, i1) {
      %c0 = arith.constant 0 : index
      cbit.store %seed, %reg[%c0] : !cbit.reg<2>
      %choice = qco.if %condition args() -> i1 {
        qco.yield %seed : i1
      } else args() {
        qco.yield %condition : i1
      }
      %value = cbit.load %reg[%c0] : !cbit.reg<2>
      return %value, %choice : i1, i1
    }
  })mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned = cast<func::ReturnOp>(function.front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(1));
}

TEST_F(ClassicalSimplificationTest,
       StoresInvalidatePotentiallyAliasedArguments) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @test(%a: !cbit.reg<2>, %b: !cbit.reg<2>, %seed: i1, %other: i1)
        -> (i1, i1) {
      %c0 = arith.constant 0 : index
      cbit.store %seed, %a[%c0] : !cbit.reg<2>
      %before = cbit.load %a[%c0] : !cbit.reg<2>
      cbit.store %other, %b[%c0] : !cbit.reg<2>
      %after = cbit.load %a[%c0] : !cbit.reg<2>
      return %before, %after : i1, i1
    }
  })mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned = cast<func::ReturnOp>(function.front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(2));
  EXPECT_TRUE(returned.getOperand(1).getDefiningOp<cbit::LoadOp>());
}

TEST_F(ClassicalSimplificationTest, WholeWritesInvalidateAliasedArguments) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @test(%a: !cbit.reg<2>, %b: !cbit.reg<2>, %seed: i1, %other: i2)
        -> i1 {
      %c0 = arith.constant 0 : index
      cbit.store %seed, %a[%c0] : !cbit.reg<2>
      cbit.write %other, %b : i2, !cbit.reg<2>
      %after = cbit.load %a[%c0] : !cbit.reg<2>
      return %after : i1
    }
  })mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned = cast<func::ReturnOp>(function.front().getTerminator());
  EXPECT_TRUE(returned.getOperand(0).getDefiningOp<cbit::LoadOp>());
}

TEST_F(ClassicalSimplificationTest, UnknownEffectsInvalidateStoredValues) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func private @touch()
    func.func @test(%reg: !cbit.reg<2>, %seed: i1) -> i1 {
      %c0 = arith.constant 0 : index
      cbit.store %seed, %reg[%c0] : !cbit.reg<2>
      func.call @touch() : () -> ()
      %value = cbit.load %reg[%c0] : !cbit.reg<2>
      return %value : i1
    }
  })mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned = cast<func::ReturnOp>(function.front().getTerminator());
  EXPECT_TRUE(returned.getOperand(0).getDefiningOp<cbit::LoadOp>());
}

TEST_F(ClassicalSimplificationTest, NestedWritesInvalidateStoredValues) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @test(%a: !cbit.reg<2>, %b: !cbit.reg<2>, %seed: i1, %condition: i1)
        -> i1 {
      %c0 = arith.constant 0 : index
      cbit.store %seed, %a[%c0] : !cbit.reg<2>
      qco.if %condition args() -> () {
        cbit.store %condition, %b[%c0] : !cbit.reg<2>
        qco.yield
      } else args() {
        qco.yield
      }
      %value = cbit.load %a[%c0] : !cbit.reg<2>
      return %value : i1
    }
  })mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned = cast<func::ReturnOp>(function.front().getTerminator());
  EXPECT_TRUE(returned.getOperand(0).getDefiningOp<cbit::LoadOp>());
}

} // namespace
