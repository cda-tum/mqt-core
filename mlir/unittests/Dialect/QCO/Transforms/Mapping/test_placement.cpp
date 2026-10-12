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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
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
constexpr std::array TYPES{PatternType::FourCycle, PatternType::Star,
                           PatternType::QubitPair};

Edges getEdges(const Graph& graph) {
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

Edges matchEdges(PatternType type, ArrayRef<size_t> vertices) {
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

class PlacementAnalysisTest : public testing::Test {
protected:
  MLIRContext context;

  void SetUp() override {
    context
        .loadDialect<QCODialect, qtensor::QTensorDialect, arith::ArithDialect,
                     cf::ControlFlowDialect, func::FuncDialect>();
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

TEST_F(PlacementAnalysisTest, AnalyzeNestedRegionSeparately) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%condition: i1, %a: !qco.qubit, %b: !qco.qubit)
        -> (!qco.qubit, !qco.qubit) {
      %x, %y = qco.if %condition args(%qa = %a, %qb = %b)
          -> (!qco.qubit, !qco.qubit) {
        %u, %v = qco.swap %qa, %qb : !qco.qubit, !qco.qubit
        qco.yield %u, %v : !qco.qubit, !qco.qubit
      } else args(%qa = %a, %qb = %b) {
        qco.yield %qa, %qb : !qco.qubit, !qco.qubit
      }
      return %x, %y : !qco.qubit, !qco.qubit
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  ScopedDiagnosticHandler handler(&context,
                                  [](Diagnostic&) { return success(); });
  EXPECT_TRUE(failed(buildConnectivityGraph(func.getBody())));
  auto branch = *func.getOps<IfOp>().begin();
  auto connectivity = buildConnectivityGraph(branch->getRegion(0));
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 1}}));
  EXPECT_EQ(connectivity->qubits[0].root, branch->getRegion(0).getArgument(0));
}

TEST_F(PlacementAnalysisTest, RejectUnsupportedConnectivity) {
  for (const auto source : {
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
           })mlir"}) {
    auto moduleOp = parse(source);
    ASSERT_TRUE(moduleOp);
    auto func =
        *llvm::find_if(moduleOp->getOps<func::FuncOp>(),
                       [](func::FuncOp op) { return !op.isExternal(); });
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic&) { return success(); });
    EXPECT_TRUE(failed(buildConnectivityGraph(func.getBody())));
  }
}

TEST(TopologyPatternMatching, ExhaustiveFiveVertexGraphs) {
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
    for (const auto type : TYPES) {
      auto permutation = nodes;
      std::optional<SmallVector<size_t, 4>> expected;
      do {
        const auto candidateEdges = matchEdges(type, permutation);
        if (std::includes(edges.begin(), edges.end(), candidateEdges.begin(),
                          candidateEdges.end())) {
          expected = SmallVector<size_t, 4>(
              permutation.begin(),
              permutation.begin() + (type == PatternType::QubitPair ? 2 : 4));
          break;
        }
      } while (std::next_permutation(permutation.begin(), permutation.end()));
      EXPECT_EQ(matchTopologyPattern(graph, type), expected);
      EXPECT_EQ(matchTopologyPattern(reversed, type), expected);
    }
    auto slices = decomposeConnectivityGraph(graph, TYPES);
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

TEST(TopologyPatternMatching, PatternAvailabilityAndPreference) {
  Graph graph({0, 1, 2, 3, 4});
  EXPECT_TRUE(succeeded(decomposeConnectivityGraph(graph, {})));
  EXPECT_FALSE(matchTopologyPattern(Graph({}), PatternType::QubitPair));
  for (const auto& [a, b] : Edges{{0, 1}, {1, 2}, {2, 3}, {0, 3}, {0, 2}}) {
    graph.addEdge(a, b);
    graph.addEdge(b, a);
  }
  EXPECT_TRUE(
      failed(decomposeConnectivityGraph(graph, {PatternType::FourCycle})));
  auto slices = decomposeConnectivityGraph(graph, TYPES);
  ASSERT_TRUE(succeeded(slices));
  ASSERT_EQ(slices->size(), 2);
  EXPECT_EQ((*slices)[0].type, PatternType::FourCycle);
  EXPECT_EQ((*slices)[1].type, PatternType::QubitPair);
  EXPECT_EQ((*slices)[1].vertices, (SmallVector<size_t, 4>{0, 2}));
}
