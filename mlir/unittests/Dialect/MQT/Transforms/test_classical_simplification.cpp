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
/// Tests for classical memory promotion and scoped condition simplification.

#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"

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
#include "llvm/ADT/STLExtras.h"

#include <bitset>
#include <map>
#include <string>

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

TEST_F(ClassicalSimplificationTest, PromotesConditionallyInitializedBits) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%condition: i1) -> !cbit.reg<1> {
        %c0 = arith.constant 0 : index
        %true = arith.constant true
        %local = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<1>
        %out = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        scf.if %condition {
          cbit.store %true, %local[%c0] : !cbit.reg<1>
        }
        scf.if %condition {
          %bit = cbit.load %local[%c0] : !cbit.reg<1>
          cbit.store %bit, %out[%c0] : !cbit.reg<1>
        }
        return %out : !cbit.reg<1>
      }
    }
  )mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  PassManager manager(&context_);
  manager.addPass(createSROA());
  manager.addPass(createMem2Reg());
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(llvm::hasSingleElement(function.getOps<cbit::AllocOp>()));
  Builder builder(&context_);
  for (bool condition : {false, true}) {
    const qco::DDArgumentBindings bindings{
        {function.getArgument(0), builder.getBoolAttr(condition)},
    };
    auto counts = qco::sample(function, 1, 1, bindings);
    ASSERT_TRUE(succeeded(counts));
    EXPECT_EQ(*counts,
              (std::map<std::string, size_t>{{condition ? "1" : "0", 1}}));
  }
}

TEST_F(ClassicalSimplificationTest,
       PromotesSnapshotsAcrossBranchesAndNestedLoops) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%condition: i1, %count: index) -> !cbit.reg<5> {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %true = arith.constant true
        %false = arith.constant false
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %out = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "out"} : !cbit.reg<5>
        %q = qco.static 0 : !qco.qubit
        %branch:2 = qco.if %condition args(%arg = %q) -> (index, !qco.qubit) {
          cbit.store %true, %reg[%c0] : !cbit.reg<2>
          %x = qco.x %arg : !qco.qubit
          qco.yield %count, %x : index, !qco.qubit
        } else args(%arg = %q) {
          qco.yield %count, %arg : index, !qco.qubit
        }
        %switched = qco.index_switch %count -> !qco.qubit
        case 0 args(%arg = %branch#1) {
          cbit.store %true, %reg[%c1] : !cbit.reg<2>
          qco.yield %arg : !qco.qubit
        }
        default args(%arg = %branch#1) {
          cbit.store %false, %reg[%c1] : !cbit.reg<2>
          qco.yield %arg : !qco.qubit
        }
        %snapshot = cbit.read %reg : !cbit.reg<2> -> i2
        scf.for %i = %c0 to %branch#0 step %c1 {
          scf.for %j = %c0 to %branch#0 step %c1 {
            %old = cbit.load %reg[%c0] : !cbit.reg<2>
            %new = arith.xori %old, %true : i1
            cbit.store %new, %reg[%c0] : !cbit.reg<2>
          }
        }
        %final = cbit.read %reg : !cbit.reg<2> -> i2
        %before = arith.extui %snapshot : i2 to i5
        %after = arith.extui %final : i2 to i5
        %two = arith.constant 2 : i5
        %shifted = arith.shli %after, %two : i5
        %packed = arith.ori %before, %shifted : i5
        cbit.write %packed, %out : i5, !cbit.reg<5>
        %qout, %measured = qco.measure %switched : !qco.qubit
        cbit.store %measured, %out[%c4] : !cbit.reg<5>
        qco.sink %qout : !qco.qubit
        return %out : !cbit.reg<5>
      }
    }
  )mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  const auto check = [&] {
    ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
    auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
    Builder builder(&context_);
    for (bool condition : {false, true}) {
      for (int64_t count : {0, 1, 2, 3}) {
        const unsigned before =
            static_cast<unsigned>(condition) | (count == 0 ? 2U : 0U);
        const unsigned after = before ^ static_cast<unsigned>(count % 2);
        const auto expected =
            std::bitset<5>(before | (after << 2U) |
                           (static_cast<unsigned>(condition) << 4U))
                .to_string();
        const qco::DDArgumentBindings bindings{
            {function.getArgument(0), builder.getBoolAttr(condition)},
            {function.getArgument(1), builder.getIndexAttr(count)},
        };
        auto counts = qco::sample(function, 1, 1, bindings);
        ASSERT_TRUE(succeeded(counts));
        EXPECT_EQ(*counts, (std::map<std::string, size_t>{{expected, 1}}));
      }
    }
  };
  check();
  PassManager manager(&context_);
  manager.addPass(createSROA());
  manager.addPass(createMem2Reg());
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  moduleOp->walk([](cbit::AllocOp alloc) {
    EXPECT_EQ(alloc.getType().getWidth(), 5);
    EXPECT_EQ(alloc->getAttr("mqt.register_name"),
              StringAttr::get(alloc.getContext(), "out"));
  });
  check();
}

} // namespace
