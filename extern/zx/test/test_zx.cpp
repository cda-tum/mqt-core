#include <gtest/gtest.h>
//#include "..Hexagram.hpp"
#include "../include/Simplify.hpp"
#include "../include/ZXDiagram.hpp"
#include "Definitions.hpp"
#include "dd/Definitions.hpp"

class RationalTest : public ::testing::Test {};

TEST_F(RationalTest, normalize) {
  zx::Rational r(-33, 16);
  EXPECT_EQ(r, zx::Rational(-1, 16));
}

TEST_F(RationalTest, from_double) {
  zx::Rational r(-dd::PI / 8);
  EXPECT_EQ(r, zx::Rational(-1, 8));
}

TEST_F(RationalTest, from_double_2) {
  zx::Rational r(-3*dd::PI / 4);
  EXPECT_EQ(r, zx::Rational(-3, 4));
}

TEST_F(RationalTest, from_double_3) {
  zx::Rational r(-7*dd::PI / 8);
  EXPECT_EQ(r, zx::Rational(-7, 8));
}

TEST_F(RationalTest, from_double_4) {
  zx::Rational r(-1*dd::PI / 32);
  EXPECT_EQ(r, zx::Rational(-1, 32));
}

TEST_F(RationalTest, from_double_5) {
  zx::Rational r(5000*dd::PI + dd::PI/4);
  EXPECT_EQ(r, zx::Rational(1, 4));
}

TEST_F(RationalTest, from_double_6) {
  zx::Rational r(-5000*dd::PI +5*dd::PI/4);
  EXPECT_EQ(r, zx::Rational(-3, 4));
}

TEST_F(RationalTest, add) {
  zx::Rational r0(1, 8);
  zx::Rational r1(7, 8);
  auto r = r0 + r1;

  EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, add_2) {
  zx::Rational r0(9, 8);
  zx::Rational r1(7, 8);
  auto r = r0 + r1;

  EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, sub) {
  zx::Rational r0(9, 8);
  zx::Rational r1(-7, 8);
  auto r = r0 - r1;

  EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, sub_2) {
  zx::Rational r0(-1, 2);
  zx::Rational r1(1, 2);
  auto r = r0 - r1;

  EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, mul) {
  zx::Rational r0(1, 8);
  zx::Rational r1(1, 2);
  auto r = r0 * r1;

  EXPECT_EQ(r, zx::Rational(1, 16));
}

TEST_F(RationalTest, mul_2) {
  zx::Rational r0(1, 8);
  zx::Rational r1(0, 1);
  auto r = r0 * r1;

  EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, div) {
  zx::Rational r0(1, 2);
  zx::Rational r1(1, 2);
  auto r = r0 / r1;

  EXPECT_EQ(r, 1);
}

class ZXDiagramTest : public ::testing::Test {
public:
  zx::ZXDiagram diag;

  /*
   * Diagram should look like this:
   * {0: (Boundary, Phase=0)} ---H--- {2: (Z, Phase=0)} ---- {3: (Z, Phase = 0)}
   * ----- {5: (Boundary, Phase = 0)}
   *                                                                 |
   *                                                                 |
   *                                                                 |
   * {1: (Boundary, Phase=0)} ------------------------------ {4: (X, Phase = 0)}
   * ----- {6: (Boundary, Phase = 0)}
   */
protected:
  virtual void SetUp() {
    diag = zx::ZXDiagram("./test/circuits/bell_state.qasm");
  }
};

TEST_F(ZXDiagramTest, parse_qasm) {
  EXPECT_EQ(diag.get_nvertices(), 7);
  EXPECT_EQ(diag.get_nedges(), 6);

  auto inputs = diag.get_inputs();
  EXPECT_EQ(inputs[0], 0);
  EXPECT_EQ(inputs[1], 1);

  auto outputs = diag.get_outputs();
  EXPECT_EQ(outputs[0], 5);
  EXPECT_EQ(outputs[1], 6);

  EXPECT_EQ(diag.get_edge(0, 2).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.get_edge(3, 4).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(1, 4).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(6, 4).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(2, 3).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(3, 5).value().type, zx::EdgeType::Simple);

  EXPECT_EQ(diag.get_vdata(0).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(1).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(2).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(3).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(4).value().type, zx::VertexType::X);
  EXPECT_EQ(diag.get_vdata(5).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(6).value().type, zx::VertexType::Boundary);

  for (auto i = 0; i < diag.get_nvertices(); i++)
    EXPECT_EQ(diag.get_vdata(6).value().phase, 0);
}

TEST_F(ZXDiagramTest, deletions) {
  diag.remove_vertex(3);
  EXPECT_EQ(diag.get_nvertices(), 6);
  EXPECT_EQ(diag.get_nedges(), 3);
  EXPECT_FALSE(diag.get_vdata(3).has_value());

  diag.remove_edge(0, 2);
  EXPECT_EQ(diag.get_nvertices(), 6);
  EXPECT_EQ(diag.get_nedges(), 2);
}

TEST_F(ZXDiagramTest, graph_like) {
  diag.to_graph_like();

  EXPECT_EQ(diag.get_edge(0, 2).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.get_edge(3, 4).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.get_edge(1, 4).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.get_edge(6, 4).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.get_edge(2, 3).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(3, 5).value().type, zx::EdgeType::Simple);

  EXPECT_EQ(diag.get_vdata(0).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(1).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(2).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(3).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(4).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(5).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(6).value().type, zx::VertexType::Boundary);
  for (auto i = 0; i < diag.get_nvertices(); i++)
    EXPECT_EQ(diag.get_vdata(6).value().phase, 0);
}

TEST_F(ZXDiagramTest, concat) {
  auto copy = diag;
  diag.concat(copy);

  ASSERT_EQ(diag.get_nedges(), 10);
  ASSERT_EQ(diag.get_nvertices(), 10);

  EXPECT_EQ(diag.get_edge(0, 2).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.get_edge(3, 4).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(1, 4).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(2, 3).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(3, 7).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.get_edge(4, 9).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(7, 8).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(8, 10).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(9, 11).value().type, zx::EdgeType::Simple);

  EXPECT_EQ(diag.get_vdata(0).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(1).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(2).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(3).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(4).value().type, zx::VertexType::X);
  EXPECT_EQ(diag.get_vdata(7).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(8).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(9).value().type, zx::VertexType::X);
  EXPECT_EQ(diag.get_vdata(10).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(11).value().type, zx::VertexType::Boundary);

  EXPECT_TRUE(diag.is_deleted(5));
  EXPECT_TRUE(diag.is_deleted(6));
}

TEST_F(ZXDiagramTest, adjoint) {
  diag = diag.adjoint();

  EXPECT_EQ(diag.get_edge(0, 2).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.get_edge(3, 4).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(1, 4).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(6, 4).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(2, 3).value().type, zx::EdgeType::Simple);
  EXPECT_EQ(diag.get_edge(3, 5).value().type, zx::EdgeType::Simple);

  EXPECT_EQ(diag.get_vdata(0).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(1).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(2).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(3).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(4).value().type, zx::VertexType::X);
  EXPECT_EQ(diag.get_vdata(5).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(6).value().type, zx::VertexType::Boundary);
}

class SimplifyTest : public ::testing::Test {
public:
protected:
  virtual void SetUp() {}
};

zx::ZXDiagram make_identity_diagram(int32_t nqubits,
                                    int32_t spiders_per_qubit) {
  zx::ZXDiagram diag(nqubits);
  std::vector<zx::Vertex> rightmost_vertices = diag.get_inputs();

  for (auto i = 0; i < nqubits; i++)
    diag.remove_edge(i, i + nqubits);

  // add identity spiders
  for (auto qubit = 0; qubit < nqubits; qubit++) {
    for (auto j = 0; j < spiders_per_qubit; j++) {
      zx::Vertex v = diag.add_vertex(qubit);
      diag.add_edge(rightmost_vertices[qubit], v);
      rightmost_vertices[qubit] = v;
    }
  }

  for (auto qubit = 0; qubit < nqubits; qubit++) {
    diag.add_edge(rightmost_vertices[qubit], qubit + nqubits);
  }

  return diag;
}

zx::ZXDiagram make_empty_diagram(int32_t nqubits) {
  auto diag = make_identity_diagram(nqubits, 0);
  for (auto i = 0; i < nqubits; i++) {
    diag.remove_edge(i, i + nqubits);
  }
  return diag;
}

TEST_F(SimplifyTest, id_simp) {
  int32_t nqubits = 3;
  int32_t spiders = 100;
  zx::ZXDiagram diag = make_identity_diagram(nqubits, spiders);

  int32_t removed = zx::id_simp(diag);

  EXPECT_EQ(removed, nqubits * spiders);
  EXPECT_EQ(diag.get_nvertices(), nqubits * 2);
  EXPECT_EQ(diag.get_nedges(), nqubits);
}

TEST_F(SimplifyTest, id_simp_2) {

  int32_t nqubits = 2;
  int32_t spiders = 100;
  zx::ZXDiagram diag = make_identity_diagram(nqubits, spiders);

  diag.add_edge(50, 150); // make vertices 50 and 150 non-removable

  int32_t removed = zx::id_simp(diag);
  EXPECT_EQ(removed, nqubits * 100 - 2);
  EXPECT_EQ(diag.get_nvertices(), nqubits * 2 + 2);
  EXPECT_EQ(diag.get_nedges(), 5);
}

TEST_F(SimplifyTest, spider_fusion) {
  int32_t nqubits = 1;
  int32_t nspiders = 100;
  zx::ZXDiagram diag = make_identity_diagram(nqubits, nspiders);

  for (zx::Vertex v = 2; v < diag.get_nvertices(); v++)
    diag.add_phase(v, zx::Rational(1, 1));

  int32_t removed = zx::spider_simp(diag);

  EXPECT_EQ(removed, nspiders - 1);
  EXPECT_EQ(3, diag.get_nvertices());
  EXPECT_EQ(2, diag.get_nedges());
  EXPECT_EQ(diag.phase(2), 0);
}

TEST_F(SimplifyTest, spider_fusion_2) {
  int32_t nqubits = 2;
  int32_t nspiders = 5;
  zx::ZXDiagram diag = make_identity_diagram(nqubits, nspiders);

  diag.add_edge(6, 11);

  int32_t removed = zx::spider_simp(diag);

  EXPECT_EQ(removed, 9);
  EXPECT_EQ(diag.get_nvertices(), 5);
  EXPECT_EQ(diag.get_nedges(), 4);

  zx::Vertex interior = diag.incident_edges(0)[0].to;
  for (zx::Vertex v : diag.get_inputs()) {
    EXPECT_TRUE(diag.connected(v, interior));
  }
  for (zx::Vertex v : diag.get_outputs()) {
    EXPECT_TRUE(diag.connected(v, interior));
  }
}

TEST_F(SimplifyTest, spider_fusion_parallel_edges) {
  int32_t nqubits = 1;
  int32_t nspiders = 3;
  zx::ZXDiagram diag = make_identity_diagram(nqubits, nspiders);
  diag.add_edge(2, 4);
  diag.set_type(4, zx::VertexType::X);

  int32_t removed = zx::spider_simp(diag);

  EXPECT_EQ(removed, 1);
  EXPECT_EQ(diag.get_nedges(), 2);
  EXPECT_EQ(diag.incident_edges(1).size(), 1);
}

TEST_F(SimplifyTest, local_comp) {
  zx::ZXDiagram diag(2);
  diag.remove_edge(0, 2);
  diag.remove_edge(1, 3);

  diag.add_vertex(0, 0, zx::Rational(1, 2), zx::VertexType::Z); // 4
  diag.add_vertex(0, 0, zx::Rational(0, 1), zx::VertexType::Z); // 5
  diag.add_vertex(0, 0, zx::Rational(0, 1), zx::VertexType::Z); // 6
  diag.add_vertex(0, 0, zx::Rational(0, 1), zx::VertexType::Z); // 7
  diag.add_vertex(0, 0, zx::Rational(0, 1), zx::VertexType::Z); // 8

  diag.add_edge(4, 5, zx::EdgeType::Hadamard);
  diag.add_edge(4, 6, zx::EdgeType::Hadamard);
  diag.add_edge(4, 7, zx::EdgeType::Hadamard);
  diag.add_edge(4, 8, zx::EdgeType::Hadamard);

  diag.add_edge(0, 5);
  diag.add_edge(1, 6);
  diag.add_edge(2, 7);
  diag.add_edge(3, 8);

  int32_t removed = zx::local_comp_simp(diag);

  EXPECT_EQ(removed, 1);

  for (zx::Vertex v = 5; v <= 8; v++) {
    EXPECT_EQ(diag.phase(v), zx::Rational(-1, 2));
    for (zx::Vertex w = 5; w <= 8; w++) {
      if (w != v) {
        ASSERT_TRUE(diag.connected(v, w));
        EXPECT_EQ(diag.get_edge(v, w).value().type, zx::EdgeType::Hadamard);
      }
    }
  }
}

TEST_F(SimplifyTest, pivot_pauli) {
  zx::ZXDiagram diag = make_identity_diagram(2, 0);

  // remove edges between input and outputs
  diag.remove_edge(0, 2);
  diag.remove_edge(1, 3);

  diag.add_vertex(0, 0, zx::Rational(1, 1), zx::VertexType::Z); // 4
  diag.add_vertex(0, 0, zx::Rational(0, 1), zx::VertexType::Z); // 5
  diag.add_vertex(0, 0, zx::Rational(0, 1), zx::VertexType::Z); // 6
  diag.add_vertex(1, 0, zx::Rational(0, 1), zx::VertexType::Z); // 7
  diag.add_vertex(0, 0, zx::Rational(0, 1), zx::VertexType::Z); // 8
  diag.add_vertex(1, 0, zx::Rational(0, 1), zx::VertexType::Z); // 9
  diag.add_vertex(1, 0, zx::Rational(0, 1), zx::VertexType::Z); // 10

  // IO-Edges
  diag.add_edge(0, 6);
  diag.add_edge(1, 7);
  diag.add_edge(2, 8);
  diag.add_edge(3, 9);

  diag.add_edge(4, 5, zx::EdgeType::Hadamard);
  diag.add_edge(4, 6, zx::EdgeType::Hadamard);
  diag.add_edge(4, 7, zx::EdgeType::Hadamard);
  diag.add_edge(4, 10, zx::EdgeType::Hadamard);
  diag.add_edge(5, 10, zx::EdgeType::Hadamard);
  diag.add_edge(5, 8, zx::EdgeType::Hadamard);
  diag.add_edge(5, 9, zx::EdgeType::Hadamard);

  int32_t removed = zx::pivot_pauli_simp(diag);

  EXPECT_EQ(removed, 1);
  EXPECT_EQ(diag.get_nedges(), 12);
  EXPECT_EQ(diag.get_nvertices(), 9);
  EXPECT_EQ(diag.phase(8), zx::Rational(1, 1));
  EXPECT_EQ(diag.phase(9), zx::Rational(1, 1));
  EXPECT_EQ(diag.phase(10), zx::Rational(0, 1));
  EXPECT_EQ(diag.phase(6), zx::Rational(0, 1));
  EXPECT_EQ(diag.phase(7), zx::Rational(0, 1));
}

TEST_F(SimplifyTest, interior_clifford) {
  int32_t nqubits = 100;
  int32_t qubit_spiders = 100;
  zx::ZXDiagram diag = make_identity_diagram(nqubits, qubit_spiders);

  zx::interior_clifford_simp(diag);

  EXPECT_EQ(diag.get_nvertices(), nqubits * 2);
  EXPECT_EQ(diag.get_nedges(), nqubits);
  for (auto v = 0; v < nqubits; v++) {
    EXPECT_TRUE(diag.connected(diag.get_inputs()[v], diag.get_outputs()[v]));
  }
}

TEST_F(SimplifyTest, interior_clifford_2) {
  zx::ZXDiagram diag(1);
  diag.remove_edge(0, 1);

  diag.add_vertex(0, 0, zx::Rational(-1, 2), zx::VertexType::X); // 2
  diag.add_vertex(0, 0, zx::Rational(1, 2), zx::VertexType::Z);  // 3
  diag.add_vertex(0, 0, zx::Rational(-1, 2), zx::VertexType::X); // 4

  diag.add_edge(2, 3);
  diag.add_edge(3, 4);
  diag.add_edge(2, 4, zx::EdgeType::Hadamard);

  diag.add_edge(0, 2);
  diag.add_edge(4, 1);

  diag.to_graph_like();
  zx::interior_clifford_simp(diag);

  EXPECT_EQ(diag.get_nvertices(), 4);
  EXPECT_EQ(diag.get_nedges(), 2);
  EXPECT_FALSE(diag.is_deleted(2));
  EXPECT_FALSE(diag.is_deleted(4));
  EXPECT_TRUE(diag.is_deleted(3));
}

TEST_F(SimplifyTest, non_pauli_pivot) {
  zx::ZXDiagram diag(1);
  diag.remove_edge(0, 1);

  diag.add_vertex(0, 0, zx::Rational(1, 4)); // 2
  diag.add_vertex(0);                        // 3
  diag.add_vertex(0);                        // 4

  diag.add_edge(0, 2);
  diag.add_edge(2, 3, zx::EdgeType::Hadamard);
  diag.add_edge(3, 4, zx::EdgeType::Hadamard);
  diag.add_edge(4, 1);

  diag.to_graph_like();
  auto res = zx::pivot_simp(diag);

  EXPECT_GT(res, 0);
  ASSERT_EQ(diag.get_nedges(), 5);
  ASSERT_EQ(diag.get_nvertices(), 6);

  EXPECT_TRUE(diag.connected(0, 7));
  EXPECT_TRUE(diag.connected(7, 4));
  EXPECT_TRUE(diag.connected(1, 4));
  EXPECT_TRUE(diag.connected(6, 4));
  EXPECT_TRUE(diag.connected(5, 6));
}

TEST_F(SimplifyTest, clifford) {
  zx::ZXDiagram diag("./test/circuits/clifford_identity_simple.qasm");
  diag.to_graph_like();
  zx::clifford_simp(diag);

  EXPECT_TRUE(diag.connected(diag.get_inputs()[0], diag.get_outputs()[0]));
}

TEST_F(SimplifyTest, clifford_2) {
  zx::ZXDiagram diag("./test/circuits/ghz_identity.qasm");

  diag.to_graph_like();

  zx::clifford_simp(diag);

  EXPECT_TRUE(diag.connected(diag.get_inputs()[0], diag.get_outputs()[0]));
  EXPECT_TRUE(diag.connected(diag.get_inputs()[1], diag.get_outputs()[1]));
  EXPECT_TRUE(diag.connected(diag.get_inputs()[2], diag.get_outputs()[2]));
}

TEST_F(SimplifyTest, clifford_3) {
  auto diag = make_empty_diagram(2);
  diag.add_vertex(0);
  diag.add_vertex(0, 0, zx::Rational(0, 1), zx::VertexType::X);

  diag.add_vertex(0);
  diag.add_vertex(1, 0, zx::Rational(0, 1), zx::VertexType::X);

  diag.add_edge(0, 4);
  diag.add_edge(1, 5);
  diag.add_edge(4, 5);
  diag.add_edge(4, 6);
  diag.add_edge(5, 7);
  diag.add_edge(6, 7);
  diag.add_edge(6, 2);
  diag.add_edge(7, 3);

  //    zx::spider_simp(diag);
  zx::clifford_simp(diag);
  EXPECT_TRUE(diag.connected(diag.get_inputs()[0], diag.get_outputs()[0]));
  EXPECT_TRUE(diag.connected(diag.get_inputs()[1], diag.get_outputs()[1]));
}

// TEST_F(SimplifyTest, non_clifford) {
//   zx::ZXDiagram diag("./test/circuits/ctrl_phase.qasm");

//   for (auto [to, from] : diag.get_edges()) {
//     std::cout << to << "-" << from << "\n";
//   }
//   std::cout << ""
//             << "\n";

//   diag.to_graph_like();
//   zx::clifford_simp(diag);

//  for (auto [to, from] : diag.get_edges()) {
//    std::cout << to << "-" << from << "\n";
//   }
// }

TEST_F(SimplifyTest, gadget_simp) {
  zx::ZXDiagram diag = make_empty_diagram(1);

  diag.add_vertex(0);                        // 2
  diag.add_vertex(0);                        // 3
  diag.add_vertex(0);                        // 4
  diag.add_vertex(0, 0, zx::Rational(1, 1)); // 5
  diag.add_vertex(0);                        // 6
  diag.add_vertex(0, 0, zx::Rational(1, 1)); // 7

  diag.add_edge(0, 2);
  diag.add_edge(3, 1);
  diag.add_edge(2, 4, zx::EdgeType::Hadamard);
  diag.add_edge(2, 6, zx::EdgeType::Hadamard);
  diag.add_edge(3, 4, zx::EdgeType::Hadamard);
  diag.add_edge(3, 6, zx::EdgeType::Hadamard);
  diag.add_edge(4, 5, zx::EdgeType::Hadamard);
  diag.add_edge(6, 7, zx::EdgeType::Hadamard);

  zx::gadget_simp(diag);

  EXPECT_TRUE(diag.connected(0, 2));
  EXPECT_TRUE(diag.connected(1, 3));
  EXPECT_TRUE(diag.connected(2, 4));
  EXPECT_TRUE(diag.connected(3, 4));
  EXPECT_TRUE(diag.connected(4, 5));
  EXPECT_EQ(diag.get_nedges(), 5);
  ASSERT_FALSE(diag.is_deleted(5));
  EXPECT_EQ(diag.phase(5), 0);
}

TEST_F(SimplifyTest, gadget_simp_2) {
  zx::ZXDiagram diag = make_empty_diagram(1);
  diag.add_vertex(0);                        // 2
  diag.add_vertex(0);                        // 3
  diag.add_vertex(0, 0, zx::Rational(1, 1)); // 4
  diag.add_vertex(0);                        // 5
  diag.add_vertex(0, 0, zx::Rational(1, 1)); // 6

  diag.add_edge(0, 2);
  diag.add_edge(2, 1);
  diag.add_edge(2, 3, zx::EdgeType::Hadamard);
  diag.add_edge(2, 5, zx::EdgeType::Hadamard);
  diag.add_edge(3, 4, zx::EdgeType::Hadamard);
  diag.add_edge(5, 6, zx::EdgeType::Hadamard);

  zx::gadget_simp(diag);

  ASSERT_FALSE(diag.is_deleted(2));
  EXPECT_TRUE(diag.connected(0, 2));
  EXPECT_TRUE(diag.connected(2, 1));
  EXPECT_EQ(diag.get_nedges(), 2);
  EXPECT_EQ(diag.phase(2), 0);
}

TEST_F(SimplifyTest, pivot_gadget_simp) {}
TEST_F(SimplifyTest, full_reduce) {
  zx::ZXDiagram diag("./test/circuits/ctrl_phase.qasm");

  zx::full_reduce(diag);

  EXPECT_TRUE(diag.is_identity());
}

TEST_F(SimplifyTest, full_reduce_2) {
  zx::ZXDiagram diag = make_empty_diagram(2);

  diag.add_vertex(0, 0, zx::Rational(1, 32), zx::VertexType::X);  // 4
  diag.add_vertex(0, 0, zx::Rational(0, 1), zx::VertexType::Z);   // 5
  diag.add_vertex(1, 0, zx::Rational(0, 1), zx::VertexType::X);   // 6
  diag.add_vertex(0, 0, zx::Rational(0, 1), zx::VertexType::Z);   // 7
  diag.add_vertex(1, 0, zx::Rational(0, 1), zx::VertexType::X);   // 8
  diag.add_vertex(0, 0, zx::Rational(-1, 32), zx::VertexType::X); // 9

  diag.add_edge(0, 4);
  diag.add_edge(4, 5);
  diag.add_edge(1, 6);
  diag.add_edge(5, 6);
  diag.add_edge(5, 7);
  diag.add_edge(6, 8);
  diag.add_edge(7, 8);
  diag.add_edge(7, 9);
  diag.add_edge(9, 2);
  diag.add_edge(8, 3);

  zx::clifford_simp(diag);
  EXPECT_TRUE(diag.is_identity());
}

TEST_F(SimplifyTest, full_reduce_3) {
  zx::ZXDiagram diag("./test/circuits/bell_state.qasm");
  auto h = diag;
  diag.invert();
  diag.concat(h);

  zx::full_reduce(diag);

  EXPECT_TRUE(diag.is_identity());
}

TEST_F(SimplifyTest, full_reduce_4) {
  zx::ZXDiagram d0("./test/circuits/C17_204_o0.qasm");
  zx::ZXDiagram d1("./test/circuits/C17_204_o1.qasm");

  d0.invert();
  d0.concat(d1);

  zx::full_reduce(d0);

  for (auto [to, from] : d0.get_edges()) {
    std::cout << to
              << (d0.get_edge(from, to).value().type == zx::EdgeType::Hadamard
                      ? "- -"
                      : "--")
              << from << " | " << from
              << (d0.get_edge(to, from).value().type == zx::EdgeType::Hadamard
                      ? "- -"
                      : "--")
              << to << "\n";
  }
  // for(auto [v, data]: d0.get_vertices())
  // std::cout << v << " p: " <<data.phase << "\n";
  // std::cout << ""
  //           << "\n";
  // for (auto [v, data] : d0.get_vertices()) {
  //   std::cout << v << " p:" << data.phase << " boundary "
  //             << (data.type == zx::VertexType::Boundary ? "True" : "False")
  //             << " type " << (d0.type(v) == zx::VertexType::Z ? "Z" : "X")
  //             << "\n";
  // }
  for (int i = 0; i < d0.get_inputs().size(); i++) {
    std::cout << d0.get_inputs()[i] << "--" << d0.get_outputs()[i] << "\n";
  }

  // std::cout << "" << "\n";

  std::cout << d0.get_nvertices() << "\n";
  std::cout << d0.get_nedges() << "\n";

  EXPECT_TRUE(d0.is_identity());
}

TEST_F(SimplifyTest, full_reduce_5) {
  zx::ZXDiagram d0("./test/circuits/test0.qasm");
  zx::ZXDiagram d1("./test/circuits/test1.qasm");

  d0.invert();
  d0.concat(d1);

   zx::full_reduce(d0);

   EXPECT_EQ(d0.get_nedges(), 3);
   EXPECT_EQ(d0.get_nvertices(), 6);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
