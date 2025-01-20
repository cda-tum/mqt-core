/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/Expression.hpp"
#include "zx/Simplify.hpp"
#include "zx/ZXDefinitions.hpp"
#include "zx/ZXDiagram.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

using zx::fullReduceApproximate;

class SimplifyTest : public ::testing::Test {};

namespace {
zx::ZXDiagram makeIdentityDiagram(const std::size_t nqubits,
                                  const std::size_t spidersPerQubit) {
  zx::ZXDiagram diag(nqubits);
  std::vector<zx::Vertex> rightmostVertices = diag.getInputs();

  for (std::size_t i = 0; i < nqubits; ++i) {
    diag.removeEdge(i, i + nqubits);
  }

  // add identity spiders
  for (zx::Qubit qubit = 0; static_cast<std::size_t>(qubit) < nqubits;
       ++qubit) {
    for (std::size_t j = 0; j < spidersPerQubit; ++j) {
      const zx::Vertex v = diag.addVertex(qubit);
      diag.addEdge(rightmostVertices[static_cast<std::size_t>(qubit)], v);
      rightmostVertices[static_cast<std::size_t>(qubit)] = v;
    }
  }

  for (std::size_t qubit = 0; qubit < nqubits; ++qubit) {
    diag.addEdge(rightmostVertices[qubit], qubit + nqubits);
  }

  return diag;
}

zx::ZXDiagram makeEmptyDiagram(const std::size_t nqubits) {
  auto diag = ::makeIdentityDiagram(nqubits, 0);
  for (std::size_t i = 0; i < nqubits; ++i) {
    diag.removeEdge(i, i + nqubits);
  }
  return diag;
}
} // namespace

TEST_F(SimplifyTest, idSimp) {
  const std::size_t nqubits = 3U;
  const std::size_t spiders = 100U;
  zx::ZXDiagram diag = ::makeIdentityDiagram(nqubits, spiders);

  const std::size_t removed = zx::idSimp(diag);

  EXPECT_EQ(removed, nqubits * spiders);
  EXPECT_EQ(diag.getNVertices(), nqubits * 2);
  EXPECT_EQ(diag.getNEdges(), nqubits);
  EXPECT_TRUE(diag.globalPhaseIsZero());
}

TEST_F(SimplifyTest, idSimp2) {
  const std::size_t nqubits = 2U;
  const std::size_t spiders = 100U;
  zx::ZXDiagram diag = ::makeIdentityDiagram(nqubits, spiders);

  // make vertices 50 and 150 non-removable
  diag.addEdge(50, 150);

  const std::size_t removed = zx::idSimp(diag);
  EXPECT_EQ(removed, nqubits * 100 - 2);
  EXPECT_EQ(diag.getNVertices(), nqubits * 2 + 2);
  EXPECT_EQ(diag.getNEdges(), 5);
  EXPECT_TRUE(diag.globalPhaseIsZero());
}

TEST_F(SimplifyTest, spiderFusion) {
  const std::size_t nqubits = 1U;
  const std::size_t nspiders = 100U;
  zx::ZXDiagram diag = ::makeIdentityDiagram(nqubits, nspiders);

  const auto nVerts = diag.getNVertices();
  for (zx::Vertex v = 2; v < nVerts; ++v) {
    diag.addPhase(v, zx::PiExpression(zx::PiRational(1, 1)));
  }

  const std::size_t removed = zx::spiderSimp(diag);

  EXPECT_EQ(removed, nspiders - 1);
  EXPECT_EQ(3, diag.getNVertices());
  EXPECT_EQ(2, diag.getNEdges());
  EXPECT_TRUE(diag.phase(2).isZero());
  EXPECT_TRUE(diag.globalPhaseIsZero());
}

TEST_F(SimplifyTest, spiderFusion2) {
  const std::size_t nqubits = 2U;
  const std::size_t nspiders = 5U;
  zx::ZXDiagram diag = ::makeIdentityDiagram(nqubits, nspiders);

  diag.addEdge(6, 11);

  const std::size_t removed = zx::spiderSimp(diag);

  EXPECT_EQ(removed, 9);
  EXPECT_EQ(diag.getNVertices(), 5);
  EXPECT_EQ(diag.getNEdges(), 4);

  const zx::Vertex interior = diag.incidentEdges(0)[0].to;
  for (const zx::Vertex v : diag.getInputs()) {
    EXPECT_TRUE(diag.connected(v, interior));
  }
  for (const zx::Vertex v : diag.getOutputs()) {
    EXPECT_TRUE(diag.connected(v, interior));
  }
  EXPECT_TRUE(diag.globalPhaseIsZero());
}

TEST_F(SimplifyTest, spiderFusionParallelEdges) {
  const std::size_t nqubits = 1U;
  const std::size_t nspiders = 3U;
  zx::ZXDiagram diag = ::makeIdentityDiagram(nqubits, nspiders);
  diag.addEdge(2, 4);
  diag.setType(4, zx::VertexType::X);

  const std::size_t removed = zx::spiderSimp(diag);

  EXPECT_EQ(removed, 1);
  EXPECT_EQ(diag.getNEdges(), 2);
  EXPECT_EQ(diag.incidentEdges(1).size(), 1);
  EXPECT_TRUE(diag.globalPhaseIsZero());
}

TEST_F(SimplifyTest, localComp) {
  zx::ZXDiagram diag(2);
  diag.removeEdge(0, 2);
  diag.removeEdge(1, 3);

  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 2)),
                 zx::VertexType::Z); // 4
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 5
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 6
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 7
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 8

  diag.addEdge(4, 5, zx::EdgeType::Hadamard);
  diag.addEdge(4, 6, zx::EdgeType::Hadamard);
  diag.addEdge(4, 7, zx::EdgeType::Hadamard);
  diag.addEdge(4, 8, zx::EdgeType::Hadamard);

  diag.addEdge(0, 5);
  diag.addEdge(1, 6);
  diag.addEdge(2, 7);
  diag.addEdge(3, 8);

  const std::size_t removed = zx::localCompSimp(diag);

  EXPECT_EQ(removed, 1);

  for (zx::Vertex v = 5; v <= 8; ++v) {
    EXPECT_TRUE(diag.phase(v) == zx::PiExpression(zx::PiRational(-1, 2)));
    for (zx::Vertex w = 5; w <= 8; ++w) {
      if (w != v) {
        ASSERT_TRUE(diag.connected(v, w));
        const auto& edge = diag.getEdge(v, w);
        const auto hasValue = edge.has_value();
        ASSERT_TRUE(hasValue);
        if (hasValue) {
          EXPECT_EQ(edge->type, zx::EdgeType::Hadamard);
        }
      }
    }
  }
}

TEST_F(SimplifyTest, pivotPauli) {
  zx::ZXDiagram diag = ::makeIdentityDiagram(2, 0);

  // remove edges between input and outputs
  diag.removeEdge(0, 2);
  diag.removeEdge(1, 3);

  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 1)),
                 zx::VertexType::Z); // 4
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 5
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 6
  diag.addVertex(1, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 7
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 8
  diag.addVertex(1, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 9
  diag.addVertex(1, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 10

  // IO-Edges
  diag.addEdge(0, 6);
  diag.addEdge(1, 7);
  diag.addEdge(2, 8);
  diag.addEdge(3, 9);

  diag.addEdge(4, 5, zx::EdgeType::Hadamard);
  diag.addEdge(4, 6, zx::EdgeType::Hadamard);
  diag.addEdge(4, 7, zx::EdgeType::Hadamard);
  diag.addEdge(4, 10, zx::EdgeType::Hadamard);
  diag.addEdge(5, 10, zx::EdgeType::Hadamard);
  diag.addEdge(5, 8, zx::EdgeType::Hadamard);
  diag.addEdge(5, 9, zx::EdgeType::Hadamard);

  const std::size_t removed = zx::pivotPauliSimp(diag);

  EXPECT_EQ(removed, 1);
  EXPECT_EQ(diag.getNEdges(), 12);
  EXPECT_EQ(diag.getNVertices(), 9);
  EXPECT_TRUE(diag.phase(8) == zx::PiExpression(zx::PiRational(1, 1)));
  EXPECT_TRUE(diag.phase(9) == zx::PiExpression(zx::PiRational(1, 1)));
  EXPECT_TRUE(diag.phase(10) == zx::PiExpression(zx::PiRational(0, 1)));
  EXPECT_TRUE(diag.phase(6) == zx::PiExpression(zx::PiRational(0, 1)));
  EXPECT_TRUE(diag.phase(7) == zx::PiExpression(zx::PiRational(0, 1)));
}

TEST_F(SimplifyTest, interiorClifford) {
  const std::size_t nqubits = 100U;
  const std::size_t qubitSpiders = 100U;
  zx::ZXDiagram diag = ::makeIdentityDiagram(nqubits, qubitSpiders);

  zx::interiorCliffordSimp(diag);

  EXPECT_EQ(diag.getNVertices(), nqubits * 2);
  EXPECT_EQ(diag.getNEdges(), nqubits);
  for (zx::Vertex v = 0; v < nqubits; ++v) {
    EXPECT_TRUE(diag.connected(diag.getInputs()[v], diag.getOutputs()[v]));
  }
  EXPECT_TRUE(diag.globalPhaseIsZero());
}

TEST_F(SimplifyTest, interiorClifford2) {
  zx::ZXDiagram diag(1);
  diag.removeEdge(0, 1);

  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(-1, 2)),
                 zx::VertexType::X); // 2
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 2)),
                 zx::VertexType::Z); // 3
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(-1, 2)),
                 zx::VertexType::X); // 4

  diag.addEdge(2, 3);
  diag.addEdge(3, 4);
  diag.addEdge(2, 4, zx::EdgeType::Hadamard);

  diag.addEdge(0, 2);
  diag.addEdge(4, 1);

  diag.toGraphlike();
  zx::interiorCliffordSimp(diag);

  EXPECT_EQ(diag.getNVertices(), 4);
  EXPECT_EQ(diag.getNEdges(), 2);
  EXPECT_FALSE(diag.isDeleted(2));
  EXPECT_FALSE(diag.isDeleted(4));
  EXPECT_TRUE(diag.isDeleted(3));
  EXPECT_FALSE(diag.globalPhaseIsZero());
  EXPECT_FALSE(diag.isIdentity());
}

TEST_F(SimplifyTest, nonPauliPivot) {
  zx::ZXDiagram diag(1);
  diag.removeEdge(0, 1);

  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 4))); // 2
  diag.addVertex(0);                                            // 3
  diag.addVertex(0);                                            // 4

  diag.addEdge(0, 2);
  diag.addEdge(2, 3, zx::EdgeType::Hadamard);
  diag.addEdge(3, 4, zx::EdgeType::Hadamard);
  diag.addEdge(4, 1);

  diag.toGraphlike();
  const auto res = zx::pivotSimp(diag);

  EXPECT_GT(res, 0);
  ASSERT_EQ(diag.getNEdges(), 5);
  ASSERT_EQ(diag.getNVertices(), 6);

  EXPECT_TRUE(diag.connected(0, 7));
  EXPECT_TRUE(diag.connected(7, 4));
  EXPECT_TRUE(diag.connected(1, 4));
  EXPECT_TRUE(diag.connected(6, 4));
  EXPECT_TRUE(diag.connected(5, 6));
  EXPECT_TRUE(diag.globalPhaseIsZero());
}

TEST_F(SimplifyTest, pauliPivot2) {
  zx::ZXDiagram diag(1);
  diag.removeEdge(0, 1);

  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 1))); // 2
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 1))); // 3
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 1))); // 4
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 1))); // 5
  diag.addEdge(0, 2);
  diag.addEdge(2, 3, zx::EdgeType::Hadamard);
  diag.addEdge(3, 4, zx::EdgeType::Hadamard);
  diag.addEdge(4, 5, zx::EdgeType::Hadamard);
  diag.addEdge(5, 1);

  diag.toGraphlike();
  const auto res = zx::pivotPauliSimp(diag);
  EXPECT_EQ(res, 1);
  EXPECT_FALSE(diag.globalPhaseIsZero());
}

TEST_F(SimplifyTest, gadgetSimp) {
  zx::ZXDiagram diag = ::makeEmptyDiagram(1);

  diag.addVertex(0);                                            // 2
  diag.addVertex(0);                                            // 3
  diag.addVertex(0);                                            // 4
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 1))); // 5
  diag.addVertex(0);                                            // 6
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 1))); // 7

  diag.addEdge(0, 2);
  diag.addEdge(3, 1);
  diag.addEdge(2, 4, zx::EdgeType::Hadamard);
  diag.addEdge(2, 6, zx::EdgeType::Hadamard);
  diag.addEdge(3, 4, zx::EdgeType::Hadamard);
  diag.addEdge(3, 6, zx::EdgeType::Hadamard);
  diag.addEdge(4, 5, zx::EdgeType::Hadamard);
  diag.addEdge(6, 7, zx::EdgeType::Hadamard);

  zx::gadgetSimp(diag);

  EXPECT_TRUE(diag.connected(0, 2));
  EXPECT_TRUE(diag.connected(1, 3));
  EXPECT_TRUE(diag.connected(2, 4));
  EXPECT_TRUE(diag.connected(3, 4));
  EXPECT_TRUE(diag.connected(4, 5));
  EXPECT_EQ(diag.getNEdges(), 5);
  ASSERT_FALSE(diag.isDeleted(5));
  EXPECT_TRUE(diag.phase(5).isZero());
  EXPECT_TRUE(diag.globalPhaseIsZero());
}

TEST_F(SimplifyTest, gadgetSimp2) {
  zx::ZXDiagram diag = ::makeEmptyDiagram(1);
  diag.addVertex(0);                                            // 2
  diag.addVertex(0);                                            // 3
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 1))); // 4
  diag.addVertex(0);                                            // 5
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 1))); // 6

  diag.addEdge(0, 2);
  diag.addEdge(2, 1);
  diag.addEdge(2, 3, zx::EdgeType::Hadamard);
  diag.addEdge(2, 5, zx::EdgeType::Hadamard);
  diag.addEdge(3, 4, zx::EdgeType::Hadamard);
  diag.addEdge(5, 6, zx::EdgeType::Hadamard);

  zx::gadgetSimp(diag);

  ASSERT_FALSE(diag.isDeleted(2));
  EXPECT_TRUE(diag.connected(0, 2));
  EXPECT_TRUE(diag.connected(2, 1));
  EXPECT_EQ(diag.getNEdges(), 2);
  EXPECT_TRUE(diag.phase(2).isZero());
  EXPECT_TRUE(diag.globalPhaseIsZero());
}

TEST_F(SimplifyTest, fullReduce2) {
  zx::ZXDiagram diag = ::makeEmptyDiagram(2);

  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 32)),
                 zx::VertexType::X); // 4
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 5
  diag.addVertex(1, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::X); // 6
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 7
  diag.addVertex(1, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::X); // 8
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(-1, 32)),
                 zx::VertexType::X); // 9

  diag.addEdge(0, 4);
  diag.addEdge(4, 5);
  diag.addEdge(1, 6);
  diag.addEdge(5, 6);
  diag.addEdge(5, 7);
  diag.addEdge(6, 8);
  diag.addEdge(7, 8);
  diag.addEdge(7, 9);
  diag.addEdge(9, 2);
  diag.addEdge(8, 3);

  zx::fullReduce(diag);
  EXPECT_TRUE(diag.isIdentity());
}

TEST_F(SimplifyTest, fullReduceApprox) {
  zx::ZXDiagram diag = ::makeEmptyDiagram(2);

  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 32)),
                 zx::VertexType::X); // 4
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 5
  diag.addVertex(1, 0, zx::PiExpression(zx::PiRational(1e-8)),
                 zx::VertexType::X); // 6
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 7
  diag.addVertex(1, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::X); // 8
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(-1, 32)),
                 zx::VertexType::X); // 9

  diag.addEdge(0, 4);
  diag.addEdge(4, 5);
  diag.addEdge(1, 6);
  diag.addEdge(5, 6);
  diag.addEdge(5, 7);
  diag.addEdge(6, 8);
  diag.addEdge(7, 8);
  diag.addEdge(7, 9);
  diag.addEdge(9, 2);
  diag.addEdge(8, 3);

  fullReduceApproximate(diag, 1e-7);
  EXPECT_TRUE(diag.isIdentity());
}

TEST_F(SimplifyTest, fullReduceNotApprox) {
  zx::ZXDiagram diag = ::makeEmptyDiagram(2);

  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(1, 32)),
                 zx::VertexType::X); // 4
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 5
  diag.addVertex(1, 0, zx::PiExpression(zx::PiRational(1e-8)),
                 zx::VertexType::X); // 6
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::Z); // 7
  diag.addVertex(1, 0, zx::PiExpression(zx::PiRational(0, 1)),
                 zx::VertexType::X); // 8
  diag.addVertex(0, 0, zx::PiExpression(zx::PiRational(-1, 32)),
                 zx::VertexType::X); // 9

  diag.addEdge(0, 4);
  diag.addEdge(4, 5);
  diag.addEdge(1, 6);
  diag.addEdge(5, 6);
  diag.addEdge(5, 7);
  diag.addEdge(6, 8);
  diag.addEdge(7, 8);
  diag.addEdge(7, 9);
  diag.addEdge(9, 2);
  diag.addEdge(8, 3);

  fullReduce(diag);
  EXPECT_FALSE(diag.isIdentity());
}

TEST_F(SimplifyTest, idSymbolic) {
  const sym::Term<double> x{sym::Variable("x")};
  const sym::Term<double> y{sym::Variable("y")};
  const sym::Term<double> z{sym::Variable("z")};

  const std::size_t nqubits = 50U;

  zx::ZXDiagram diag = ::makeIdentityDiagram(nqubits, 100);
  zx::PiExpression e;
  e += x;
  diag.setPhase((nqubits * 2) + 5, e);
  EXPECT_EQ(e.numTerms(), 1);
  EXPECT_EQ(diag.phase(nqubits * 2 + 5).numTerms(), 1);

  zx::fullReduce(diag);

  EXPECT_EQ(diag.getNVertices(), 2 * nqubits + 1);
  EXPECT_EQ(diag.getNEdges(), nqubits + 1);
  EXPECT_TRUE(diag.globalPhaseIsZero());
}

TEST_F(SimplifyTest, equivalenceSymbolic) {
  const sym::Term<double> x{sym::Variable("x")};
  const sym::Term<double> y{sym::Variable("y")};
  const sym::Term<double> z{sym::Variable("z")};
  zx::ZXDiagram d1 = ::makeEmptyDiagram(3);

  // first circuit
  d1.addVertex(0, 0); // 6
  d1.addHadamardEdge(0, 6);

  d1.addVertex(0, 0);                                        // 7
  d1.addVertex(1, 0, zx::PiExpression(), zx::VertexType::X); // 8
  d1.addEdge(7, 8);
  d1.addEdge(6, 7);
  d1.addEdge(1, 8);

  d1.addVertex(0, 0);                                        // 9
  d1.addVertex(2, 0, zx::PiExpression(), zx::VertexType::X); // 10
  d1.addEdge(9, 10);
  d1.addEdge(7, 9);
  d1.addEdge(2, 10);

  d1.addVertex(0, 0, zx::PiExpression(x), zx::VertexType::Z); // 11
  d1.addVertex(1, 0, zx::PiExpression(y), zx::VertexType::X); // 12
  d1.addVertex(2, 0, zx::PiExpression(z), zx::VertexType::X); // 13
  d1.addEdge(9, 11);
  d1.addEdge(8, 12);
  d1.addEdge(10, 13);

  d1.addVertex(0, 0);                                        // 14
  d1.addVertex(1, 0, zx::PiExpression(), zx::VertexType::X); // 15
  d1.addEdge(14, 15);
  d1.addEdge(11, 14);
  d1.addEdge(12, 15);

  d1.addVertex(0, 0);                                        // 16
  d1.addVertex(2, 0, zx::PiExpression(), zx::VertexType::X); // 17
  d1.addEdge(16, 17);
  d1.addEdge(14, 16);
  d1.addEdge(13, 17);

  d1.addVertex(0, 0); // 18
  d1.addHadamardEdge(16, 18);

  // second circuit
  d1.addVertex(0, 0); // 19
  d1.addHadamardEdge(18, 19);

  d1.addVertex(0, 0);                                        // 20
  d1.addVertex(1, 0, zx::PiExpression(), zx::VertexType::X); // 21
  d1.addEdge(20, 21);
  d1.addEdge(19, 20);
  d1.addEdge(17, 21);

  d1.addVertex(0, 0);                                        // 22
  d1.addVertex(1, 0, zx::PiExpression(), zx::VertexType::X); // 23
  d1.addEdge(22, 23);
  d1.addEdge(20, 22);
  d1.addEdge(15, 23);

  d1.addVertex(0, 0, -zx::PiExpression(x), zx::VertexType::Z); // 24
  d1.addVertex(1, 0, -zx::PiExpression(y), zx::VertexType::X); // 25
  d1.addVertex(2, 0, -zx::PiExpression(z), zx::VertexType::X); // 26
  d1.addEdge(22, 24);
  d1.addEdge(23, 25);
  d1.addEdge(21, 26);

  d1.addVertex(0, 0);                                        // 27
  d1.addVertex(1, 0, zx::PiExpression(), zx::VertexType::X); // 28
  d1.addEdge(24, 27);
  d1.addEdge(26, 28);
  d1.addEdge(28, 27);

  d1.addVertex(0, 0);                                        // 29
  d1.addVertex(1, 0, zx::PiExpression(), zx::VertexType::X); // 30
  d1.addEdge(29, 30);
  d1.addEdge(27, 29);
  d1.addEdge(25, 30);

  d1.addHadamardEdge(29, 3);
  d1.addEdge(30, 4);
  d1.addEdge(28, 5);

  zx::fullReduce(d1);

  EXPECT_EQ(d1.getNEdges(), 3);
  EXPECT_EQ(d1.getNVertices(), 6);
  EXPECT_TRUE(d1.isIdentity());
}

TEST_F(SimplifyTest, OnlyDeletedVertices) {
  // This is a regression test. The following code should not throw an
  // exception. It previously did because the code did not handle the case where
  // the diagram only contains deleted vertices.
  zx::ZXDiagram diag = ::makeIdentityDiagram(1, 0);
  diag.makeAncilla(0);
  // The following simplifies the diagram to the empty diagram.
  zx::fullReduce(diag);
  EXPECT_EQ(diag.getNEdges(), 0);
  EXPECT_EQ(diag.getNVertices(), 0);
  // A subsequent simplification should not throw an exception.
  EXPECT_NO_THROW(zx::fullReduce(diag););
}
