/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Support/Passes.h"

#include "ExactUnitaryTest.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"

#include <cstddef>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {
class CoalesceControlledGatesTest : public testing::Test {
protected:
  MLIRContext context;

  void SetUp() override {
    context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  }

  static size_t countControls(ModuleOp moduleOp) {
    size_t count = 0;
    moduleOp.walk([&](CtrlOp) { ++count; });
    return count;
  }

  void coalesce(ModuleOp moduleOp, bool cleanup = true) {
    ASSERT_TRUE(succeeded(verify(moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp)));
    PassManager pm(&context);
    pm.addPass(createCoalesceControlledGates());
    if (cleanup) {
      pm.addPass(createCanonicalizerPass());
    }
    ASSERT_TRUE(succeeded(pm.run(moduleOp)));
    ASSERT_TRUE(succeeded(verify(moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp)));
  }

  void checkInverse(Value (QCOProgramBuilder::*gate)(Value),
                    Value (QCOProgramBuilder::*inverse)(Value)) {
    auto moduleOp =
        QCOProgramBuilder::build(&context, [&](QCOProgramBuilder& builder) {
          auto [control, target] =
              builder.ctrl(builder.staticQubit(0), builder.staticQubit(1),
                           [&](Value q) { return (builder.*gate)(q); });
          builder.ctrl(control, target,
                       [&](Value q) { return (builder.*inverse)(q); });
          return SmallVector<Value>{};
        });
    ASSERT_TRUE(moduleOp);
    OwningOpRef<ModuleOp> original(moduleOp->clone());
    ASSERT_NO_FATAL_FAILURE(coalesce(*moduleOp));
    EXPECT_EQ(countControls(*moduleOp), 0U);
    ::mqt::test::expectFullUnitaryEqual(*original, *moduleOp, 2);
  }
};
} // namespace

TEST_F(CoalesceControlledGatesTest, DefaultPipelineMergesParameterizedBodies) {
  auto moduleOp =
      QCOProgramBuilder::build(&context, [](QCOProgramBuilder& builder) {
        auto control = builder.staticQubit(0);
        auto target = builder.staticQubit(1);
        auto [c, t] = builder.ctrl(control, target,
                                   [&](Value q) { return builder.rz(0.2, q); });
        builder.ctrl(c, t, [&](Value q) { return builder.rz(0.3, q); });
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  OwningOpRef<ModuleOp> original(moduleOp->clone());
  PassManager pm(&context);
  populateDefaultQCOOptimizationPipeline(pm);
  pm.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));
  EXPECT_EQ(countControls(*moduleOp), 1U);
  moduleOp->walk([](CtrlOp op) { EXPECT_EQ(op.getNumBodyUnitaries(), 1U); });
  ::mqt::test::expectFullUnitaryEqual(*original, *moduleOp, 2);
}

TEST_F(CoalesceControlledGatesTest, ReusesNamedInverseCanonicalization) {
  checkInverse(&QCOProgramBuilder::h, &QCOProgramBuilder::h);
  checkInverse(&QCOProgramBuilder::x, &QCOProgramBuilder::x);
  checkInverse(&QCOProgramBuilder::y, &QCOProgramBuilder::y);
  checkInverse(&QCOProgramBuilder::z, &QCOProgramBuilder::z);
  checkInverse(&QCOProgramBuilder::s, &QCOProgramBuilder::sdg);
  checkInverse(&QCOProgramBuilder::sdg, &QCOProgramBuilder::s);
  checkInverse(&QCOProgramBuilder::t, &QCOProgramBuilder::tdg);
  checkInverse(&QCOProgramBuilder::tdg, &QCOProgramBuilder::t);
  checkInverse(&QCOProgramBuilder::sx, &QCOProgramBuilder::sxdg);
  checkInverse(&QCOProgramBuilder::sxdg, &QCOProgramBuilder::sx);
}

TEST_F(CoalesceControlledGatesTest, MultiTargetBodiesRetainControlledPhase) {
  auto moduleOp =
      QCOProgramBuilder::build(&context, [](QCOProgramBuilder& builder) {
        auto control = builder.staticQubit(0);
        SmallVector<Value> targets{
            builder.staticQubit(1),
            builder.staticQubit(2),
        };
        for (size_t i = 0; i < 2; ++i) {
          auto [controls, result] =
              builder.ctrl(ValueRange{control}, targets, [&](ValueRange args) {
                builder.gphase(0.125);
                return SmallVector<Value>{
                    builder.h(args[0]),
                    builder.x(args[1]),
                };
              });
          control = controls.front();
          targets.assign(result.begin(), result.end());
        }
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  OwningOpRef<ModuleOp> original(moduleOp->clone());
  ASSERT_NO_FATAL_FAILURE(coalesce(*moduleOp));
  EXPECT_EQ(countControls(*moduleOp), 1U);
  moduleOp->walk([](HOp) { ADD_FAILURE() << "uncancelled Hadamard"; });
  moduleOp->walk([](XOp) { ADD_FAILURE() << "uncancelled X"; });
  ::mqt::test::expectFullUnitaryEqual(*original, *moduleOp, 3);
}

TEST_F(CoalesceControlledGatesTest,
       RespectsOrderPartitionAndQuantumBoundaries) {
  for (const auto* boundary : {
           "control order",
           "target order",
           "partition",
           "target gate",
           "control gate",
           "disjoint gate",
           "barrier",
       }) {
    SCOPED_TRACE(boundary);
    auto moduleOp =
        QCOProgramBuilder::build(&context, [&](QCOProgramBuilder& builder) {
          SmallVector<Value> wires;
          for (size_t i = 0; i < 5; ++i) {
            wires.push_back(builder.staticQubit(i));
          }
          const auto body = [&](ValueRange args) {
            SmallVector<Value> outputs;
            for (auto q : args) {
              outputs.push_back(builder.h(q));
            }
            return outputs;
          };
          auto [controlsOut, targetsOut] =
              builder.ctrl(ValueRange(wires).take_front(2),
                           ValueRange(wires).slice(2, 2), body);
          llvm::copy(controlsOut, wires.begin());
          llvm::copy(targetsOut, wires.begin() + 2);
          const StringRef kind(boundary);
          if (kind == "control order") {
            std::swap(wires[0], wires[1]);
          } else if (kind == "target order") {
            std::swap(wires[2], wires[3]);
          } else if (kind == "target gate") {
            wires[2] = builder.x(wires[2]);
          } else if (kind == "control gate") {
            wires[0] = builder.x(wires[0]);
          } else if (kind == "disjoint gate") {
            wires[4] = builder.x(wires[4]);
          } else if (kind == "barrier") {
            wires[4] = builder.barrier(ValueRange{wires[4]}).front();
          }
          const size_t controls = kind == "partition" ? 1 : 2;
          builder.ctrl(ValueRange(wires).take_front(controls),
                       ValueRange(wires).slice(controls, 4 - controls), body);
          return SmallVector<Value>{};
        });
    ASSERT_TRUE(moduleOp);
    OwningOpRef<ModuleOp> original(moduleOp->clone());
    ASSERT_NO_FATAL_FAILURE(coalesce(*moduleOp, false));
    EXPECT_EQ(countControls(*moduleOp), 2U);
    ::mqt::test::expectFullUnitaryEqual(*original, *moduleOp, 5);
  }
}

TEST_F(CoalesceControlledGatesTest, RuntimeAnglesKeepDominance) {
  Value parameter;
  auto moduleOp =
      QCOProgramBuilder::build(&context, [&](QCOProgramBuilder& builder) {
        parameter = builder.floatConstant(0.2);
        auto [control, target] =
            builder.ctrl(builder.staticQubit(0), builder.staticQubit(1),
                         [&](Value q) { return builder.rz(parameter, q); });
        auto opposite = arith::NegFOp::create(builder, parameter);
        builder.ctrl(control, target,
                     [&](Value q) { return builder.rz(opposite, q); });
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  ASSERT_TRUE(succeeded(function.insertArgument(0, Float64Type::get(&context),
                                                {}, function.getLoc())));
  parameter.replaceAllUsesWith(function.getArgument(0));
  OwningOpRef<ModuleOp> original(moduleOp->clone());
  PassManager pm(&context);
  populateDefaultQCOOptimizationPipeline(pm);
  pm.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));
  EXPECT_LE(countControls(*moduleOp), 1U);
  for (const double theta : {0.73, -2.9, 6.28}) {
    OwningOpRef<ModuleOp> expected(original->clone());
    OwningOpRef<ModuleOp> actual(moduleOp->clone());
    for (auto mod : {expected.get(), actual.get()}) {
      auto func = *mod.getOps<func::FuncOp>().begin();
      OpBuilder builder(&func.getBody().front(),
                        func.getBody().front().begin());
      auto value = arith::ConstantOp::create(builder, func.getLoc(),
                                             builder.getF64FloatAttr(theta));
      func.getArgument(0).replaceAllUsesWith(value);
      PassManager fold(&context);
      fold.addPass(createCanonicalizerPass());
      ASSERT_TRUE(succeeded(fold.run(mod)));
    }
    ::mqt::test::expectFullUnitaryEqual(*expected, *actual, 2);
  }
}
