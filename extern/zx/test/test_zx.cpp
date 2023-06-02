#include "Definitions.hpp"
#include "Rational.hpp"
#include "Simplify.hpp"
#include "ZXDiagram.hpp"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

class ZXDiagramTest : public ::testing::Test {
public:
  zx::ZXDiagram diag;

  /*
   * Diagram should look like this:
   * {0: (Boundary, Phase=0)} ---H--- {4: (Z, Phase=0)} ---- {5: (Z, Phase = 0)}
   * ----- {1: (Boundary, Phase = 0)}
   *                                                                 |
   *                                                                 |
   *                                                                 |
   * {2: (Boundary, Phase=0)} ------------------------------ {6: (X, Phase = 0)}
   * ----- {3: (Boundary, Phase = 0)}
   */
protected:
  void SetUp() override {
    diag = zx::ZXDiagram();
    diag.addQubits(2);
    diag.addVertex(0, 0, zx::PiExpression(), zx::VertexType::Z);
    diag.addEdge(0, 4, zx::EdgeType::Hadamard);
    diag.addVertex(0, 0, zx::PiExpression(), zx::VertexType::Z);
    diag.addEdge(4, 5);
    diag.addVertex(0, 0, zx::PiExpression(), zx::VertexType::X);
    diag.addEdge(2, 6);
    diag.addEdge(5, 6);
    diag.addEdge(5, 1);
    diag.addEdge(6, 3);
  }
};

TEST_F(ZXDiagramTest, createDiagram) {
  EXPECT_EQ(diag.getNVertices(), 7);
  EXPECT_EQ(diag.getNEdges(), 6);

  auto inputs = diag.getInputs();
  EXPECT_EQ(inputs[0], 0);
  EXPECT_EQ(inputs[1], 2);

  auto outputs = diag.getOutputs();
  EXPECT_EQ(outputs[0], 1);
  EXPECT_EQ(outputs[1], 3);

  EXPECT_EQ(diag.getEdge(0, 4).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.getEdge(4, 5).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.getEdge(2, 6).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.getEdge(5, 6).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.getEdge(5, 1).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.getEdge(6, 3).value().type, zx::EdgeType::Simple);

  EXPECT_EQ(diag.getVData(0).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(1).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(2).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(3).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(4).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.getVData(5).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.getVData(6).value().type, zx::VertexType::X);

  const auto nVerts = diag.getNVertices();
  for (std::size_t i = 0; i < nVerts; ++i) {
    EXPECT_TRUE(diag.getVData(6).value().phase.isZero());
  }
}

TEST_F(ZXDiagramTest, deletions) {
  diag.removeVertex(5);
  EXPECT_EQ(diag.getNVertices(), 6);
  EXPECT_EQ(diag.getNEdges(), 3);
  EXPECT_FALSE(diag.getVData(5).has_value());

  diag.removeEdge(0, 4);
  EXPECT_EQ(diag.getNVertices(), 6);
  EXPECT_EQ(diag.getNEdges(), 2);
}

TEST_F(ZXDiagramTest, graphLike) {
  diag.toGraphlike();

  EXPECT_EQ(diag.getEdge(0, 4).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.getEdge(5, 6).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.getEdge(2, 6).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.getEdge(3, 6).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.getEdge(4, 5).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.getEdge(5, 1).value().type, zx::EdgeType::Simple);

  EXPECT_EQ(diag.getVData(0).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(1).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(2).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(3).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(4).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.getVData(5).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.getVData(6).value().type, zx::VertexType::Z);

  const auto nVerts = diag.getNVertices();
  for (std::size_t i = 0; i < nVerts; ++i) {
    EXPECT_TRUE(diag.getVData(i).value().phase.isZero());
  }
}

TEST_F(ZXDiagramTest, adjoint) {
  diag = diag.adjoint();

  EXPECT_EQ(diag.getEdge(0, 4).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.getEdge(5, 6).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.getEdge(2, 6).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.getEdge(3, 6).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.getEdge(4, 5).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.getEdge(5, 1).value().type, zx::EdgeType::Simple);

  EXPECT_EQ(diag.getVData(0).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(1).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(2).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(3).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.getVData(4).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.getVData(5).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.getVData(6).value().type, zx::VertexType::X);
}

TEST_F(ZXDiagramTest, approximate) {
  zx::ZXDiagram almostId(3);

  almostId.removeEdge(0, 3);
  const auto v =
      almostId.addVertex(0, 1, zx::PiExpression(zx::PiRational(1e-8)));
  almostId.addEdge(0, v);
  almostId.addEdge(v, 3);

  EXPECT_FALSE(almostId.phase(v).isZero());

  almostId.approximateCliffords(1e-7);

  EXPECT_TRUE(almostId.phase(v).isZero());
}

TEST_F(ZXDiagramTest, ancilla) {
  zx::ZXDiagram cx(2);
  cx.removeEdge(0, 2);
  cx.removeEdge(1, 3);
  const auto tar = cx.addVertex(0, 0, zx::PiExpression{}, zx::VertexType::X);

  const auto ctrl = cx.addVertex(1);

  cx.addEdge(tar, ctrl);
  cx.addEdge(0, tar);
  cx.addEdge(tar, 2);
  cx.addEdge(1, ctrl);
  cx.addEdge(ctrl, 3);
  EXPECT_EQ(cx.getInputs().size(), 2);
  EXPECT_EQ(cx.getOutputs().size(), 2);
  EXPECT_EQ(cx.getNVertices(), 6);

  cx.makeAncilla(1);
  EXPECT_EQ(cx.getInputs().size(), 1);
  EXPECT_EQ(cx.getOutputs().size(), 1);
  EXPECT_EQ(cx.getNVertices(), 6);

  zx::fullReduce(cx);

  EXPECT_EQ(cx.getNEdges(), 1);
  for (const auto& [v, data] : cx.getVertices()) {
    std::cout << v << " " << (data.type == zx::VertexType::Boundary) << "\n";
  }
  EXPECT_EQ(cx.getNVertices(), 2);
  EXPECT_TRUE(cx.isIdentity());
}

TEST_F(ZXDiagramTest, RemoveScalarSubDiagram) {
  zx::ZXDiagram idWithScal(1);

  const auto v = idWithScal.addVertex(1);
  const auto w = idWithScal.addVertex(2);
  idWithScal.addEdge(v, w);

  zx::fullReduce(idWithScal);

  EXPECT_EQ(idWithScal.getNVertices(), 2);
  EXPECT_EQ(idWithScal.getNEdges(), 1);
  EXPECT_TRUE(idWithScal.isDeleted(v));
  EXPECT_TRUE(idWithScal.isDeleted(w));
}

TEST_F(ZXDiagramTest, AdjMat) {
  zx::ZXDiagram diag(3);

  const auto& adj = diag.getAdjMat();

  for (const auto& [v, _] : diag.getVertices()) {
    for (const auto& [w, _] : diag.getVertices()) {
      if (diag.connected(v, w) || v == w) {
        EXPECT_TRUE(adj[v][w]);
        EXPECT_TRUE(adj[w][v]);
      } else {
        EXPECT_FALSE(adj[v][w]);
        EXPECT_FALSE(adj[w][v]);
      }
    }
  }
}

TEST_F(ZXDiagramTest, ConnectedSet) {
  zx::ZXDiagram diag(3);
  auto          connected = diag.getConnectedSet(diag.getInputs());

  for (const auto& v : connected) {
    EXPECT_TRUE(diag.isIn(v, diag.getOutputs()));
  }

  connected = diag.getConnectedSet(diag.getInputs(), {4});

  EXPECT_TRUE(diag.isIn(3, connected));
  EXPECT_FALSE(diag.isIn(4, connected));
  EXPECT_TRUE(diag.isIn(5, connected));
}
