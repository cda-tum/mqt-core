/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Mapping/Placement.h"
#include "mqt/Dialect/QCO/Utils/Graph.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/AnalysisManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {
using PatternType = CompilerTarget::TopologyPatterns::Type;
using Edges = std::set<std::pair<size_t, size_t>>;
constexpr std::array SMALL_PATTERN_TYPES{
    PatternType::FourCycle,
    PatternType::QubitPair,
    PatternType::Star,
};
constexpr std::array EXTENDED_PATTERN_TYPES{
    PatternType::FourCycle,  PatternType::QubitPair, PatternType::Star,
    PatternType::Chain4,     PatternType::Chain6,    PatternType::SixCycle,
    PatternType::EightCycle, PatternType::Grid2x3,   PatternType::Grid2x4,
};
class TopologyPatternMatching : public testing::Test {
protected:
  const CompilerTarget target = llvm::cantFail(
      CompilerTarget::create(4,
                             CompilerTarget::Connectivity::fromCouplings(
                                 {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {0, 2}}),
                             CompilerTarget::NativeOperations::unrestricted()));
  const ArrayRef<CompilerTarget::TopologyPatterns::Group> patterns =
      *target.topologyPatterns();
};
} // namespace

static Edges getEdges(const Graph& graph) {
  Edges edges;
  for (const auto a : graph.getNodes()) {
    for (const auto b : graph.getNeighbours(a)) {
      if (a < b) {
        edges.emplace(a, b);
      }
    }
  }
  return edges;
}

static Edges matchEdges(PatternType type, ArrayRef<size_t> vertices) {
  Edges edges;
  const auto add = [&](size_t a, size_t b) {
    edges.insert(std::minmax(vertices[a], vertices[b]));
  };
  if (type == PatternType::FourCycle) {
    for (size_t i = 0; i < 4; ++i) {
      add(i, (i + 1) % 4);
    }
  } else {
    for (size_t i = 1; i < (type == PatternType::Star ? 4U : 2U); ++i) {
      add(0, i);
    }
  }
  return edges;
}

namespace {
class PlacementAnalysisTest : public testing::Test {
protected:
  MLIRContext context;

  void SetUp() override {
    context.loadDialect<QCODialect, qtensor::QTensorDialect,
                        arith::ArithDialect, cf::ControlFlowDialect,
                        func::FuncDialect, scf::SCFDialect>();
  }

  OwningOpRef<ModuleOp> parse(StringRef source) {
    auto moduleOp = parseSourceString<ModuleOp>(source, &context);
    EXPECT_TRUE(moduleOp);
    if (moduleOp) {
      EXPECT_TRUE(succeeded(verify(*moduleOp)));
      EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
    }
    return moduleOp;
  }
};
} // namespace

TEST_F(PlacementAnalysisTest, ScalarWireIdentityAndRepeatedInteractions) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%a: !qco.qubit, %b: !qco.qubit)
        -> (!qco.qubit, !qco.qubit) {
      %idle = qco.alloc : !qco.qubit
      %c = qco.alloc : !qco.qubit
      %a1 = qco.h %a : !qco.qubit
      %a2, %b1 = qco.ctrl(%a1) targets(%target = %b) {
        %out = qco.x %target : !qco.qubit
        qco.yield %out : !qco.qubit
      } : ({!qco.qubit}, {!qco.qubit})
      %b2, %a3 = qco.swap %b1, %a2 : !qco.qubit, !qco.qubit
      %a4, %bit = qco.measure %a3 : !qco.qubit
      %a5 = qco.reset %a4 : !qco.qubit
      %a6, %c1 = qco.swap %a5, %c : !qco.qubit, !qco.qubit
      %b3, %c2, %idle1 = qco.barrier %b2, %c1, %idle
          : !qco.qubit, !qco.qubit, !qco.qubit
      qco.sink %c2 : !qco.qubit
      qco.sink %idle1 : !qco.qubit
      return %a6, %b3 : !qco.qubit, !qco.qubit
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  moduleOp->print(beforeStream);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto connectivity = buildConnectivityGraph(func.getBody());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(connectivity->qubits.size(), 4);
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 1}, {0, 3}}));
  EXPECT_EQ(connectivity->graph.getDegree(2), 0);
  EXPECT_EQ(connectivity->graph.getDegree(0), 2);
  EXPECT_EQ(connectivity->qubits[0].root, func.getArgument(0));
  EXPECT_FALSE(connectivity->qubits[0].slot);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  moduleOp->print(afterStream);
  EXPECT_EQ(before, after);
}

TEST_F(PlacementAnalysisTest, TensorSlotsAndFromElementsKeepWireIdentity) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%q: tensor<3x!qco.qubit>) -> tensor<3x!qco.qubit> {
      %i0 = arith.constant 0 : index
      %i2 = arith.constant 2 : index
      %t1, %a = qtensor.extract %q[%i2] : tensor<3x!qco.qubit>
      %t2, %b = qtensor.extract %t1[%i0] : tensor<3x!qco.qubit>
      %c = qco.alloc : !qco.qubit
      %a1, %c1 = qco.swap %a, %c : !qco.qubit, !qco.qubit
      %t3 = qtensor.insert %a1 into %t2[%i2] : tensor<3x!qco.qubit>
      %t4, %a2 = qtensor.extract %t3[%i2] : tensor<3x!qco.qubit>
      %b1, %a3 = qco.swap %b, %a2 : !qco.qubit, !qco.qubit
      %t5 = qtensor.insert %a3 into %t4[%i2] : tensor<3x!qco.qubit>
      %t6 = qtensor.insert %b1 into %t5[%i0] : tensor<3x!qco.qubit>
      %other = qtensor.alloc(%i2) : tensor<2x!qco.qubit>
      qtensor.dealloc %other : tensor<2x!qco.qubit>
      %packed = qtensor.from_elements %c1 : tensor<1x!qco.qubit>
      %p1, %c2 = qtensor.extract %packed[%i0] : tensor<1x!qco.qubit>
      %c3 = qco.h %c2 : !qco.qubit
      %p2 = qtensor.insert %c3 into %p1[%i0] : tensor<1x!qco.qubit>
      qtensor.dealloc %p2 : tensor<1x!qco.qubit>
      return %t6 : tensor<3x!qco.qubit>
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto connectivity = buildConnectivityGraph(func.getBody());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(connectivity->qubits.size(), 6);
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 2}, {2, 3}}));
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(connectivity->qubits[i].root, func.getArgument(0));
    EXPECT_EQ(connectivity->qubits[i].slot, i);
  }
  EXPECT_EQ(connectivity->graph.getDegree(1), 0);
}

TEST_F(PlacementAnalysisTest, NestedControlFlowUnionsInteractions) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%condition: i1, %n: index, %a: !qco.qubit,
                      %b: !qco.qubit, %c: !qco.qubit)
        -> (!qco.qubit, !qco.qubit, !qco.qubit) {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %x, %y, %z = scf.for %i = %zero to %n step %one
          iter_args(%fa = %a, %fb = %b, %fc = %c)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %u, %v, %w = qco.if %condition args(%qa = %fa, %qb = %fb, %qc = %fc)
            -> (!qco.qubit, !qco.qubit, !qco.qubit) {
          %sa, %sb = qco.swap %qa, %qb : !qco.qubit, !qco.qubit
          qco.yield %sa, %sb, %qc : !qco.qubit, !qco.qubit, !qco.qubit
        } else args(%qa = %fa, %qb = %fb, %qc = %fc) {
          %flag, %wa, %wb, %wc = scf.while
              (%state = %n, %aa = %qa, %ab = %qb, %ac = %qc)
              : (index, !qco.qubit, !qco.qubit, !qco.qubit)
              -> (i1, !qco.qubit, !qco.qubit, !qco.qubit) {
            %ha = qco.h %aa : !qco.qubit
            scf.condition(%condition) %condition, %ha, %ab, %ac
                : i1, !qco.qubit, !qco.qubit, !qco.qubit
          } do {
          ^bb0(%flag_arg: i1, %ba: !qco.qubit, %bb: !qco.qubit, %bc: !qco.qubit):
            %sb, %sc = qco.swap %bb, %bc : !qco.qubit, !qco.qubit
            scf.yield %n, %ba, %sb, %sc
                : index, !qco.qubit, !qco.qubit, !qco.qubit
          }
          qco.yield %wa, %wb, %wc : !qco.qubit, !qco.qubit, !qco.qubit
        }
        scf.yield %u, %v, %w : !qco.qubit, !qco.qubit, !qco.qubit
      }
      %r, %s, %t = qco.index_switch %n
          -> (!qco.qubit, !qco.qubit, !qco.qubit)
      case 0 args(%qa = %x, %qb = %y, %qc = %z) {
        %sa, %sc = qco.swap %qa, %qc : !qco.qubit, !qco.qubit
        qco.yield %sa, %qb, %sc : !qco.qubit, !qco.qubit, !qco.qubit
      }
      default args(%qa = %x, %qb = %y, %qc = %z) {
        qco.yield %qa, %qb, %qc : !qco.qubit, !qco.qubit, !qco.qubit
      }
      return %r, %s, %t : !qco.qubit, !qco.qubit, !qco.qubit
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto connectivity = buildConnectivityGraph(func.getBody());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(connectivity->qubits.size(), 3);
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 1}, {1, 2}, {0, 2}}));
  EXPECT_EQ(connectivity->qubits[0].root, func.getArgument(2));
  auto loop = *func.getOps<scf::ForOp>().begin();
  auto branch = *loop.getBody()->getOps<IfOp>().begin();
  connectivity = buildConnectivityGraph(branch.getThenRegion());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 1}}));
  EXPECT_EQ(connectivity->qubits[0].root,
            branch.getThenRegion().getArgument(0));
}

TEST_F(PlacementAnalysisTest, TensorSlotsAcrossControlFlow) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%q: tensor<3x!qco.qubit>, %n: index)
        -> tensor<3x!qco.qubit> {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %result = scf.for %i = %zero to %n step %one
          iter_args(%qs = %q) -> tensor<3x!qco.qubit> {
        %t1, %a = qtensor.extract %qs[%zero] : tensor<3x!qco.qubit>
        %t2, %b = qtensor.extract %t1[%two] : tensor<3x!qco.qubit>
        %x, %y = qco.swap %a, %b : !qco.qubit, !qco.qubit
        %t3 = qtensor.insert %x into %t2[%zero] : tensor<3x!qco.qubit>
        %t4 = qtensor.insert %y into %t3[%two] : tensor<3x!qco.qubit>
        scf.yield %t4 : tensor<3x!qco.qubit>
      }
      %t5, %c = qtensor.extract %result[%one] : tensor<3x!qco.qubit>
      %t6, %d = qtensor.extract %t5[%zero] : tensor<3x!qco.qubit>
      %u, %v = qco.swap %c, %d : !qco.qubit, !qco.qubit
      %t7 = qtensor.insert %u into %t6[%one] : tensor<3x!qco.qubit>
      %t8 = qtensor.insert %v into %t7[%zero] : tensor<3x!qco.qubit>
      return %t8 : tensor<3x!qco.qubit>
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto connectivity = buildConnectivityGraph(func.getBody());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(connectivity->qubits.size(), 3);
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 1}, {0, 2}}));
  EXPECT_EQ(connectivity->qubits[2].slot, 2);
}

TEST_F(PlacementAnalysisTest, RegionInterfaceTracksProducedAndReorderedQubits) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit() {
      %x, %y = scf.execute_region -> (!qco.qubit, !qco.qubit) {
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
        %u, %v = qco.swap %a, %b : !qco.qubit, !qco.qubit
        scf.yield %v, %u : !qco.qubit, !qco.qubit
      }
      %c = qco.alloc : !qco.qubit
      %z, %w = qco.swap %x, %c : !qco.qubit, !qco.qubit
      qco.sink %y : !qco.qubit
      qco.sink %z : !qco.qubit
      qco.sink %w : !qco.qubit
      return
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto connectivity = buildConnectivityGraph(func.getBody());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(connectivity->qubits.size(), 3);
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 1}, {1, 2}}));
}

TEST_F(PlacementAnalysisTest, RejectTensorPermutationAtLoopBackedge) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%q: tensor<2x!qco.qubit>, %n: index)
        -> tensor<2x!qco.qubit> {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %result = scf.for %i = %zero to %n step %one
          iter_args(%qs = %q) -> tensor<2x!qco.qubit> {
        %t1, %a = qtensor.extract %qs[%zero] : tensor<2x!qco.qubit>
        %t2, %b = qtensor.extract %t1[%one] : tensor<2x!qco.qubit>
        %t3 = qtensor.insert %a into %t2[%one] : tensor<2x!qco.qubit>
        %t4 = qtensor.insert %b into %t3[%zero] : tensor<2x!qco.qubit>
        scf.yield %t4 : tensor<2x!qco.qubit>
      }
      return %result : tensor<2x!qco.qubit>
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& message) {
    diagnostic = message.str();
    return success();
  });
  EXPECT_TRUE(failed(buildConnectivityGraph(func.getBody())));
  EXPECT_NE(diagnostic.find("backedges"), std::string::npos);
}

TEST_F(PlacementAnalysisTest, CachesAndInvalidatesConnectivity) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%a: !qco.qubit, %b: !qco.qubit)
        -> (!qco.qubit, !qco.qubit) {
      %x, %y = qco.swap %a, %b : !qco.qubit, !qco.qubit
      return %x, %y : !qco.qubit, !qco.qubit
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  ModuleAnalysisManager manager(moduleOp.get(), nullptr);
  auto analyses = AnalysisManager(manager).nest(func);
  EXPECT_FALSE(analyses.getCachedAnalysis<RegionConnectivityAnalysis>());
  const auto& initial = analyses.getAnalysis<RegionConnectivityAnalysis>();
  ASSERT_TRUE(succeeded(initial.getConnectivity()));
  EXPECT_EQ(getEdges(initial.getConnectivity()->graph), (Edges{{0, 1}}));
  EXPECT_EQ(&initial, &analyses.getAnalysis<RegionConnectivityAnalysis>());
  AnalysisManager::PreservedAnalyses preserved;
  preserved.preserve<RegionConnectivityAnalysis>();
  analyses.invalidate(preserved);
  ASSERT_TRUE(analyses.getCachedAnalysis<RegionConnectivityAnalysis>());
  EXPECT_EQ(&initial, &analyses.getAnalysis<RegionConnectivityAnalysis>());

  auto swap = *func.getOps<SWAPOp>().begin();
  swap->replaceAllUsesWith(swap->getOperands());
  swap.erase();
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));
  analyses.invalidate(AnalysisManager::PreservedAnalyses{});
  EXPECT_FALSE(analyses.getCachedAnalysis<RegionConnectivityAnalysis>());
  const auto& updated = analyses.getAnalysis<RegionConnectivityAnalysis>();
  ASSERT_TRUE(succeeded(updated.getConnectivity()));
  EXPECT_TRUE(getEdges(updated.getConnectivity()->graph).empty());
}

TEST_F(PlacementAnalysisTest, RejectTensorPermutationAtRegionBoundary) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%q: tensor<2x!qco.qubit>, %condition: i1)
        -> tensor<2x!qco.qubit> {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %result = qco.if %condition args(%qs = %q) -> tensor<2x!qco.qubit> {
        %t1, %a = qtensor.extract %qs[%zero] : tensor<2x!qco.qubit>
        %t2, %b = qtensor.extract %t1[%one] : tensor<2x!qco.qubit>
        %t3 = qtensor.insert %a into %t2[%one] : tensor<2x!qco.qubit>
        %t4 = qtensor.insert %b into %t3[%zero] : tensor<2x!qco.qubit>
        qco.yield %t4 : tensor<2x!qco.qubit>
      } else args(%qs = %q) {
        qco.yield %qs : tensor<2x!qco.qubit>
      }
      return %result : tensor<2x!qco.qubit>
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& message) {
    diagnostic = message.str();
    return success();
  });
  EXPECT_TRUE(failed(buildConnectivityGraph(func.getBody())));
  EXPECT_NE(diagnostic.find("tensor-slot identities at region joins"),
            std::string::npos);
  auto branch = *func.getOps<IfOp>().begin();
  EXPECT_TRUE(failed(RegionConnectivityAnalysis(branch).getConnectivity()));
  auto connectivity = buildConnectivityGraph(branch->getRegion(0));
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_TRUE(getEdges(connectivity->graph).empty());
}

TEST_F(PlacementAnalysisTest, RejectUnsupportedConnectivity) {
  for (const auto* source : {
           R"mlir(func.func @dynamic_size(%n: index) {
             %q = qtensor.alloc(%n) : tensor<?x!qco.qubit>
             qtensor.dealloc %q : tensor<?x!qco.qubit>
             return
           })mlir",
           R"mlir(func.func @dynamic_index(%q: tensor<2x!qco.qubit>, %i: index)
               -> tensor<2x!qco.qubit> {
             %t, %a = qtensor.extract %q[%i] : tensor<2x!qco.qubit>
             %out = qtensor.insert %a into %t[%i] : tensor<2x!qco.qubit>
             return %out : tensor<2x!qco.qubit>
           })mlir",
           R"mlir(func.func private @external(!qco.qubit) -> !qco.qubit
           func.func @call(%q: !qco.qubit) -> !qco.qubit {
             %out = func.call @external(%q) : (!qco.qubit) -> !qco.qubit
             return %out : !qco.qubit
           })mlir",
           R"mlir(func.func private @external(!qco.qubit) -> !qco.qubit
               attributes {mqt.unitary}
           func.func @call(%q: !qco.qubit) -> !qco.qubit {
             %out = qco.call @external(%q) : (!qco.qubit) -> !qco.qubit
             return %out : !qco.qubit
           })mlir",
           R"mlir(func.func @three_qubits(%a: !qco.qubit, %b: !qco.qubit,
                                         %c: !qco.qubit)
               -> (!qco.qubit, !qco.qubit, !qco.qubit) {
             %x, %y, %z = qco.ctrl(%a) targets(%qb = %b, %qc = %c) {
               %u, %v = qco.swap %qb, %qc
                   : !qco.qubit, !qco.qubit
               qco.yield %u, %v : !qco.qubit, !qco.qubit
             } : ({!qco.qubit}, {!qco.qubit, !qco.qubit})
             return %x, %y, %z : !qco.qubit, !qco.qubit, !qco.qubit
           })mlir",
           R"mlir(func.func @multiple_blocks() {
             cf.br ^exit
           ^exit:
             return
           })mlir",
       }) {
    auto moduleOp = parse(source);
    ASSERT_TRUE(moduleOp);
    auto func =
        *llvm::find_if(moduleOp->getOps<func::FuncOp>(),
                       [](func::FuncOp op) { return !op.isExternal(); });
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic&) { return success(); });
    EXPECT_TRUE(failed(RegionConnectivityAnalysis(func).getConnectivity()));
  }
}

TEST(ConnectivityGraph, DirectedEdgeLookup) {
  EXPECT_EQ(Graph({}).getNumNodes(), 0);
  Graph graph({0, 1});
  EXPECT_EQ(graph.getNumNodes(), 2);
  graph.addEdge(0, 1);
  EXPECT_TRUE(graph.hasEdge(0, 1));
  EXPECT_FALSE(graph.hasEdge(1, 0));
  EXPECT_FALSE(graph.hasEdge(0, 0));
  EXPECT_FALSE(graph.hasEdge(2, 0));
  graph.removeEdge(0, 1);
  EXPECT_FALSE(graph.hasEdge(0, 1));
  EXPECT_EQ(graph.getNumNodes(), 2);
  graph.addEdge(1, 2);
  EXPECT_EQ(graph.getNumNodes(), 3);
  graph.clearEdges();
  EXPECT_EQ(graph.getNumNodes(), 3);
}

TEST_F(TopologyPatternMatching, ExhaustiveFiveVertexGraphs) {
  // Sparse IDs and reversed edge insertion also exercise determinism.
  constexpr std::array<size_t, 5> nodes{2, 5, 11, 17, 23};
  for (size_t mask = 0; mask < (1U << 10U); ++mask) {
    SCOPED_TRACE(mask);
    Graph graph(nodes);
    size_t bit = 0;
    for (size_t a = 0; a < nodes.size(); ++a) {
      for (size_t b = a + 1; b < nodes.size(); ++b, ++bit) {
        if ((mask & (1U << bit)) != 0) {
          graph.addEdge(nodes[a], nodes[b]);
          graph.addEdge(nodes[b], nodes[a]);
        }
      }
    }
    const auto edges = getEdges(graph);
    Graph reversed(nodes);
    for (const auto& [a, b] : llvm::reverse(edges)) {
      reversed.addEdge(b, a);
      reversed.addEdge(a, b);
    }
    for (const auto& pattern : patterns) {
      const auto type = pattern.type();
      auto permutation = nodes;
      std::optional<SmallVector<size_t, 4>> expected;
      do {
        const auto candidateEdges = matchEdges(type, permutation);
        if (std::ranges::includes(edges, candidateEdges)) {
          expected = SmallVector<size_t, 4>(
              permutation.begin(),
              permutation.begin() + (type == PatternType::QubitPair ? 2 : 4));
          break;
        }
      } while (std::ranges::next_permutation(permutation).found);
      EXPECT_EQ(matchTopologyPattern(graph, pattern), expected);
      EXPECT_EQ(matchTopologyPattern(reversed, pattern), expected);
    }
    auto slices = decomposeConnectivityGraph(graph, patterns);
    ASSERT_TRUE(succeeded(slices));
    Edges reconstructed;
    for (const auto& slice : *slices) {
      for (const auto& edge : matchEdges(slice.type, slice.vertices)) {
        EXPECT_TRUE(reconstructed.insert(edge).second);
      }
    }
    EXPECT_EQ(reconstructed, edges);
    EXPECT_EQ(getEdges(graph), edges);
  }
}

TEST_F(TopologyPatternMatching, PatternAvailabilityAndPreference) {
  Graph graph({0, 1, 2, 3, 4});
  EXPECT_TRUE(succeeded(decomposeConnectivityGraph(
      graph, ArrayRef<CompilerTarget::TopologyPatterns::Group>{})));
  EXPECT_FALSE(matchTopologyPattern(Graph({}), patterns[1]));
  for (const auto& [a, b] : Edges{{0, 1}, {1, 2}, {2, 3}, {0, 3}, {0, 2}}) {
    graph.addEdge(a, b);
    graph.addEdge(b, a);
  }
  EXPECT_TRUE(failed(decomposeConnectivityGraph(graph, patterns.take_front())));
  const auto reversedPatterns = llvm::to_vector(llvm::reverse(patterns));
  auto slices = decomposeConnectivityGraph(graph, reversedPatterns);
  ASSERT_TRUE(succeeded(slices));
  ASSERT_EQ(slices->size(), 2);
  EXPECT_EQ((*slices)[0].type, PatternType::FourCycle);
  EXPECT_EQ((*slices)[1].type, PatternType::QubitPair);
  EXPECT_EQ((*slices)[1].vertices, (SmallVector<size_t, 4>{0, 2}));
  const auto pairTarget = llvm::cantFail(CompilerTarget::create(
      2, CompilerTarget::Connectivity::fromCouplings({{0, 1}}),
      CompilerTarget::NativeOperations::unrestricted()));
  auto pairs =
      decomposeConnectivityGraph(graph, *pairTarget.topologyPatterns());
  ASSERT_TRUE(succeeded(pairs));
  ASSERT_EQ(pairs->size(), 5);
  for (const auto& pair : *pairs) {
    EXPECT_EQ(pair.type, PatternType::QubitPair);
  }
}

TEST_F(TopologyPatternMatching, LargerPatternsReuseMatcherAndDecomposition) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  SmallVector<CompilerTarget::Coupling> couplings;
  for (size_t row = 0; row < 2; ++row) {
    for (size_t column = 0; column < 3; ++column) {
      couplings.emplace_back(row * 4 + column, row * 4 + column + 1);
    }
  }
  for (size_t column = 0; column < 4; ++column) {
    couplings.emplace_back(column, column + 4);
  }
  const auto largerTarget = llvm::cantFail(CompilerTarget::create(
      8, CompilerTarget::Connectivity::fromCouplings(couplings),
      CompilerTarget::NativeOperations::unrestricted()));
  for (const auto type : {
           PatternType::Chain4,
           PatternType::Chain6,
           PatternType::SixCycle,
           PatternType::EightCycle,
           PatternType::Grid2x3,
           PatternType::Grid2x4,
       }) {
    const auto* pattern = largerTarget.topologyPattern(type);
    ASSERT_NE(pattern, nullptr);
    ASSERT_NE(pattern->size(), 0);
    Graph graph(llvm::to_vector(llvm::seq<size_t>(0, pattern->arity())));
    for (const auto& [a, b] : pattern->edges()) {
      graph.addEdge(a, b);
      graph.addEdge(b, a);
    }
    auto match = matchTopologyPattern(graph, *pattern);
    ASSERT_TRUE(match);
    EXPECT_EQ(match->size(), pattern->arity());
    for (const auto& [a, b] : pattern->edges()) {
      EXPECT_TRUE(graph.hasEdge((*match)[a], (*match)[b]));
    }
    const std::array<const CompilerTarget::TopologyPatterns::Group*, 2> groups{
        nullptr,
        pattern,
    };
    auto slices = decomposeConnectivityGraph(graph, groups);
    ASSERT_TRUE(succeeded(slices));
    ASSERT_EQ(slices->size(), 1);
    EXPECT_EQ(slices->front().type, type);
    EXPECT_EQ(slices->front().vertices, *match);
    const auto layout =
        findTopologyPatternLayout(graph, largerTarget, context, 100000, {type});
    ASSERT_TRUE(layout);
    EXPECT_EQ(layout->size(), pattern->arity());
    EXPECT_EQ(std::set<size_t>(layout->begin(), layout->end()).size(),
              layout->size());
    for (const auto& [a, b] : pattern->edges()) {
      EXPECT_TRUE(largerTarget.areAdjacent((*layout)[a], (*layout)[b]));
    }
  }
}

namespace {
using Layouts = SmallVector<SmallVector<size_t>, 0>;
} // namespace

static Layouts bruteForceLayouts(const Graph& graph,
                                 const CompilerTarget& target) {
  auto nodes = graph.getNodes();
  llvm::sort(nodes);
  auto permutation = llvm::to_vector(llvm::seq<size_t>(0, target.numSites()));
  Layouts result;
  if (nodes.size() > permutation.size()) {
    return result;
  }
  do {
    bool valid = true;
    for (size_t a = 0; a < nodes.size(); ++a) {
      for (size_t b = a + 1; b < nodes.size(); ++b) {
        valid &= !graph.hasEdge(nodes[a], nodes[b]) ||
                 target.areAdjacent(permutation[a], permutation[b]);
      }
    }
    if (valid) {
      result.emplace_back(permutation.begin(),
                          permutation.begin() + nodes.size());
    }
  } while (std::ranges::next_permutation(permutation).found);
  llvm::sort(result);
  result.erase(std::ranges::unique(result).begin(), result.end());
  return result;
}
TEST(TopologyPatternJoin, ExhaustiveGraphsMatchBruteForce) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  // All labelled five-vertex queries include disconnected, cyclic, star and
  // impossible joins. Sparse IDs exercise row ordering independently of IDs.
  constexpr std::array<size_t, 5> nodes{2, 5, 11, 17, 23};
  for (const auto& edges : {
           SmallVector<CompilerTarget::Coupling>{
               {0, 1},
               {1, 2},
               {2, 3},
               {3, 4},
           },
           SmallVector<CompilerTarget::Coupling>{
               {0, 1},
               {0, 2},
               {0, 3},
               {0, 4},
           },
           SmallVector<CompilerTarget::Coupling>{
               {0, 1},
               {1, 2},
               {2, 3},
               {0, 3},
               {0, 2},
               {3, 4},
               {1, 4},
           },
       }) {
    const auto target = llvm::cantFail(CompilerTarget::create(
        5, CompilerTarget::Connectivity::fromCouplings(edges),
        CompilerTarget::NativeOperations::unrestricted()));
    for (size_t mask = 0; mask < (1U << 10U); ++mask) {
      SCOPED_TRACE(mask);
      Graph graph(nodes);
      size_t bit = 0;
      for (size_t a = 0; a < nodes.size(); ++a) {
        for (size_t b = a + 1; b < nodes.size(); ++b, ++bit) {
          if ((mask & (1U << bit)) != 0) {
            graph.addEdge(nodes[b], nodes[a]);
            graph.addEdge(nodes[a], nodes[b]);
          }
        }
      }
      auto layouts = enumerateTopologyLayouts(graph, target, context);
      llvm::sort(layouts);
      EXPECT_EQ(layouts, bruteForceLayouts(graph, target));
      const auto first = findTopologyLayout(graph, target, context, 100000);
      EXPECT_EQ(first.has_value(), !layouts.empty());
      if (first) {
        EXPECT_TRUE(llvm::is_contained(layouts, *first));
      }
      for (const auto selected :
           {ArrayRef(SMALL_PATTERN_TYPES), ArrayRef(EXTENDED_PATTERN_TYPES)}) {
        const auto joined =
            findTopologyPatternLayout(graph, target, context, 100000, selected);
        EXPECT_EQ(joined.has_value(), !layouts.empty());
        if (joined) {
          EXPECT_TRUE(llvm::is_contained(layouts, *joined));
        }
      }
    }
  }
}

TEST(TopologyPatternJoin, ParallelOrderAndBoundaryCases) {
  MLIRContext serial(MLIRContext::Threading::DISABLED);
  MLIRContext parallel;
  for (const auto& connectivity : {
           CompilerTarget::Connectivity::allToAll(),
           CompilerTarget::Connectivity::fromCouplings(
               {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {0, 2}, {3, 4}, {1, 4}}),
       }) {
    const auto target = llvm::cantFail(CompilerTarget::create(
        5, connectivity, CompilerTarget::NativeOperations::unrestricted()));
    for (const auto& edges : {
             Edges{},
             Edges{{2, 5}},
             Edges{{2, 5}, {11, 17}},
             Edges{{2, 5}, {5, 11}, {11, 17}, {2, 17}},
             Edges{{2, 5}, {2, 11}, {2, 17}, {17, 23}},
         }) {
      Graph graph({23, 17, 11, 5, 2});
      for (const auto& [a, b] : edges) {
        graph.addEdge(a, b);
        graph.addEdge(b, a);
      }
      auto layouts = enumerateTopologyLayouts(graph, target, parallel);
      EXPECT_EQ(layouts, enumerateTopologyLayouts(graph, target, serial));
      llvm::sort(layouts);
      EXPECT_EQ(layouts, bruteForceLayouts(graph, target));
    }
    EXPECT_EQ(enumerateTopologyLayouts(Graph({}), target, parallel),
              (Layouts{SmallVector<size_t>{}}));
    EXPECT_EQ(findTopologyLayout(Graph({}), target, parallel, 0),
              std::optional(SmallVector<size_t>{}));
    EXPECT_FALSE(findTopologyLayout(Graph({0, 1}), target, parallel, 1));
    EXPECT_FALSE(findTopologyLayout(Graph({0, 1, 2, 3, 4, 5}), target, parallel,
                                    100000));
    EXPECT_TRUE(
        enumerateTopologyLayouts(Graph({0, 1, 2, 3, 4, 5}), target, parallel)
            .empty());
    // Fewer logical qubits than target sites must not duplicate each layout
    // for permutations of the unused hardware sites.
    const Graph idle({99});
    auto layouts = enumerateTopologyLayouts(idle, target, parallel);
    llvm::sort(layouts);
    EXPECT_EQ(layouts, bruteForceLayouts(idle, target));
  }

  // Enough hardware sites to exercise parallel search, independently of the
  // small-input serial cutoff. Count two-edge paths by their middle vertex.
  SmallVector<CompilerTarget::Coupling> edges;
  for (size_t vertex = 0; vertex < 100; ++vertex) {
    if (vertex % 10 != 9) {
      edges.emplace_back(vertex, vertex + 1);
    }
    if (vertex + 10 < 100) {
      edges.emplace_back(vertex, vertex + 10);
    }
  }
  const auto target = llvm::cantFail(CompilerTarget::create(
      100, CompilerTarget::Connectivity::fromCouplings(edges),
      CompilerTarget::NativeOperations::unrestricted()));
  Graph path({5, 7, 11});
  for (const auto& [a, b] : Edges{{5, 7}, {7, 11}}) {
    path.addEdge(a, b);
    path.addEdge(b, a);
  }
  const auto layouts = enumerateTopologyLayouts(path, target, parallel);
  EXPECT_EQ(layouts, enumerateTopologyLayouts(path, target, serial));
  EXPECT_EQ(layouts.size(), 4 * 2 + 32 * 6 + 64 * 12);
  for (const size_t budget : {0U, 2U, 100000U}) {
    const auto first = findTopologyLayout(path, target, parallel, budget);
    EXPECT_EQ(first, findTopologyLayout(path, target, serial, budget));
    if (budget < 3) {
      EXPECT_FALSE(first);
    } else {
      ASSERT_TRUE(first);
      EXPECT_TRUE(llvm::is_contained(layouts, *first));
    }
  }
  for (const auto& row : layouts) {
    EXPECT_NE(row[0], row[2]);
    EXPECT_TRUE(target.areAdjacent(row[0], row[1]));
    EXPECT_TRUE(target.areAdjacent(row[1], row[2]));
  }
}

TEST(TopologyPatternJoin, SelectedPatternsRespectBudgetsAndThreadCounts) {
  SmallVector<CompilerTarget::Coupling> couplings;
  for (size_t vertex = 0; vertex < 64; ++vertex) {
    if (vertex % 8 != 7) {
      couplings.emplace_back(vertex, vertex + 1);
    }
    if (vertex + 8 < 64) {
      couplings.emplace_back(vertex, vertex + 8);
    }
  }
  const auto target = llvm::cantFail(CompilerTarget::create(
      64, CompilerTarget::Connectivity::fromCouplings(couplings),
      CompilerTarget::NativeOperations::unrestricted()));
  MLIRContext serial(MLIRContext::Threading::DISABLED);
  for (const bool cycle : {false, true}) {
    Graph graph(llvm::to_vector(llvm::seq<size_t>(0, 9)));
    for (size_t vertex = 1; vertex < 9; ++vertex) {
      graph.addEdge(vertex - 1, vertex);
      graph.addEdge(vertex, vertex - 1);
    }
    if (cycle) {
      graph.addEdge(0, 8);
      graph.addEdge(8, 0);
    }
    for (const auto selected :
         {ArrayRef(SMALL_PATTERN_TYPES), ArrayRef(EXTENDED_PATTERN_TYPES)}) {
      for (const size_t budget : {0U, 1U, 64U, 256U, 100000U}) {
        const auto expected =
            findTopologyPatternLayout(graph, target, serial, budget, selected);
        if (cycle || budget == 0) {
          // An odd cycle cannot embed in a bipartite grid.
          EXPECT_FALSE(expected);
        } else if (budget == 100000) {
          ASSERT_TRUE(expected);
        }
        if (expected) {
          EXPECT_EQ(std::set<size_t>(expected->begin(), expected->end()).size(),
                    graph.getNumNodes());
          for (const auto& [a, b] : getEdges(graph)) {
            EXPECT_TRUE(target.areAdjacent((*expected)[a], (*expected)[b]));
          }
        }
        for (const unsigned workers : {1U, 2U, 4U, 8U}) {
          SCOPED_TRACE(workers);
          SCOPED_TRACE(budget);
          llvm::StdThreadPool pool(llvm::hardware_concurrency(workers));
          MLIRContext parallel(MLIRContext::Threading::DISABLED);
          parallel.setThreadPool(pool);
          EXPECT_EQ(findTopologyPatternLayout(graph, target, parallel, budget,
                                              selected),
                    expected);
        }
      }
    }
  }
  EXPECT_EQ(findTopologyPatternLayout(Graph({}), target, serial, 0,
                                      EXTENDED_PATTERN_TYPES),
            std::optional(SmallVector<size_t>{}));
  EXPECT_FALSE(findTopologyPatternLayout(
      Graph(llvm::to_vector(llvm::seq<size_t>(0, 65))), target, serial, 100000,
      EXTENDED_PATTERN_TYPES));
}

TEST(TopologyPatternJoin, UnavailablePatternsFallBackToCompletePairs) {
  SmallVector<CompilerTarget::Coupling> couplings;
  for (size_t a = 0; a < 20; ++a) {
    for (size_t b = a + 1; b < 20; ++b) {
      couplings.emplace_back(a, b);
    }
  }
  const auto target = llvm::cantFail(CompilerTarget::create(
      20, CompilerTarget::Connectivity::fromCouplings(couplings),
      CompilerTarget::NativeOperations::unrestricted()));
  ASSERT_EQ(target.topologyPattern(PatternType::Chain6), nullptr);
  Graph graph({0, 1, 2, 3, 4, 5});
  for (size_t vertex = 1; vertex < 6; ++vertex) {
    graph.addEdge(vertex - 1, vertex);
    graph.addEdge(vertex, vertex - 1);
  }
  MLIRContext context(MLIRContext::Threading::DISABLED);
  EXPECT_FALSE(findTopologyPatternLayout(graph, target, context, 100000,
                                         {PatternType::Chain6}));
  const auto layout =
      findTopologyPatternLayout(graph, target, context, 100000,
                                {PatternType::Chain6, PatternType::QubitPair});
  ASSERT_TRUE(layout);
  EXPECT_EQ(layout->size(), graph.getNumNodes());
  EXPECT_EQ(std::set<size_t>(layout->begin(), layout->end()).size(),
            graph.getNumNodes());
  for (const auto& [a, b] : getEdges(graph)) {
    EXPECT_TRUE(target.areAdjacent((*layout)[a], (*layout)[b]));
  }
}
