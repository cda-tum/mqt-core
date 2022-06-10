#include <cstdint>
#include <gtest/gtest.h>
//#include "..Hexagram.hpp"
#include "../include/Simplify.hpp"
#include "../include/ZXDiagram.hpp"
#include "Definitions.hpp"
#include "Expression.hpp"
#include "Rational.hpp"
#include "dd/Definitions.hpp"

using zx::Expression;

class RationalTest : public ::testing::Test {};

TEST_F(RationalTest, normalize) {
  zx::PyRational r(-33, 16);
  EXPECT_EQ(r, zx::PyRational(-1, 16));
}

TEST_F(RationalTest, from_double) {
  zx::PyRational r(-dd::PI / 8);
  EXPECT_EQ(r, zx::PyRational(-1, 8));
}

TEST_F(RationalTest, from_double_2) {
  zx::PyRational r(-3 * dd::PI / 4);
  EXPECT_EQ(r, zx::PyRational(-3, 4));
}

TEST_F(RationalTest, from_double_3) {
  zx::PyRational r(-7 * dd::PI / 8);
  EXPECT_EQ(r, zx::PyRational(-7, 8));
}

TEST_F(RationalTest, from_double_4) {
  zx::PyRational r(-1 * dd::PI / 32);
  EXPECT_EQ(r, zx::PyRational(-1, 32));
}

TEST_F(RationalTest, from_double_5) {
  zx::PyRational r(5000 * dd::PI + dd::PI / 4);
  EXPECT_EQ(r, zx::PyRational(1, 4));
}

TEST_F(RationalTest, from_double_6) {
  zx::PyRational r(-5000 * dd::PI + 5 * dd::PI / 4);
  EXPECT_EQ(r, zx::PyRational(-3, 4));
}

TEST_F(RationalTest, from_double_7) {
  zx::PyRational r(0.1);
  std::cout << r << "\n";
}

TEST_F(RationalTest, add) {
  zx::PyRational r0(1, 8);
  zx::PyRational r1(7, 8);
  auto r = r0 + r1;

  EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, add_2) {
  zx::PyRational r0(9, 8);
  zx::PyRational r1(7, 8);
  auto r = r0 + r1;

  EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, sub) {
  zx::PyRational r0(9, 8);
  zx::PyRational r1(-7, 8);
  auto r = r0 - r1;

  EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, sub_2) {
  zx::PyRational r0(-1, 2);
  zx::PyRational r1(1, 2);
  auto r = r0 - r1;

  EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, mul) {
  zx::PyRational r0(1, 8);
  zx::PyRational r1(1, 2);
  auto r = r0 * r1;

  EXPECT_EQ(r, zx::PyRational(1, 16));
}

TEST_F(RationalTest, mul_2) {
  zx::PyRational r0(1, 8);
  zx::PyRational r1(0, 1);
  auto r = r0 * r1;

  EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, div) {
  zx::PyRational r0(1, 2);
  zx::PyRational r1(1, 2);
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
  virtual void SetUp() { diag = zx::ZXDiagram("circuits/bell_state.qasm"); }
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
    EXPECT_TRUE(diag.get_vdata(6).value().phase.is_zero());
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
    EXPECT_TRUE(diag.get_vdata(6).value().phase.is_zero());
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
    diag.add_phase(v, zx::Expression(zx::PyRational(1, 1)));

  int32_t removed = zx::spider_simp(diag);

  EXPECT_EQ(removed, nspiders - 1);
  EXPECT_EQ(3, diag.get_nvertices());
  EXPECT_EQ(2, diag.get_nedges());
  EXPECT_TRUE(diag.phase(2).is_zero());
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

  diag.add_vertex(0, 0, zx::PyRational(1, 2), zx::VertexType::Z); // 4
  diag.add_vertex(0, 0, zx::PyRational(0, 1), zx::VertexType::Z); // 5
  diag.add_vertex(0, 0, zx::PyRational(0, 1), zx::VertexType::Z); // 6
  diag.add_vertex(0, 0, zx::PyRational(0, 1), zx::VertexType::Z); // 7
  diag.add_vertex(0, 0, zx::PyRational(0, 1), zx::VertexType::Z); // 8

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
    EXPECT_TRUE(diag.phase(v) == zx::Expression(zx::PyRational(-1, 2)));
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

  diag.add_vertex(0, 0, zx::PyRational(1, 1), zx::VertexType::Z); // 4
  diag.add_vertex(0, 0, zx::PyRational(0, 1), zx::VertexType::Z); // 5
  diag.add_vertex(0, 0, zx::PyRational(0, 1), zx::VertexType::Z); // 6
  diag.add_vertex(1, 0, zx::PyRational(0, 1), zx::VertexType::Z); // 7
  diag.add_vertex(0, 0, zx::PyRational(0, 1), zx::VertexType::Z); // 8
  diag.add_vertex(1, 0, zx::PyRational(0, 1), zx::VertexType::Z); // 9
  diag.add_vertex(1, 0, zx::PyRational(0, 1), zx::VertexType::Z); // 10

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
  EXPECT_TRUE(diag.phase(8) == zx::Expression(zx::PyRational(1, 1)));
  EXPECT_TRUE(diag.phase(9) == zx::PyRational(1, 1));
  EXPECT_TRUE(diag.phase(10) == zx::PyRational(0, 1));
  EXPECT_TRUE(diag.phase(6) == zx::PyRational(0, 1));
  EXPECT_TRUE(diag.phase(7) == zx::PyRational(0, 1));
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

  diag.add_vertex(0, 0, zx::PyRational(-1, 2), zx::VertexType::X); // 2
  diag.add_vertex(0, 0, zx::PyRational(1, 2), zx::VertexType::Z);  // 3
  diag.add_vertex(0, 0, zx::PyRational(-1, 2), zx::VertexType::X); // 4

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

  diag.add_vertex(0, 0, zx::PyRational(1, 4)); // 2
  diag.add_vertex(0);                          // 3
  diag.add_vertex(0);                          // 4

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
  zx::ZXDiagram diag("circuits/clifford_identity_simple.qasm");
  diag.to_graph_like();
  zx::clifford_simp(diag);

  EXPECT_TRUE(diag.connected(diag.get_inputs()[0], diag.get_outputs()[0]));
}

TEST_F(SimplifyTest, clifford_2) {
  zx::ZXDiagram diag("circuits/ghz_identity.qasm");

  diag.to_graph_like();

  zx::clifford_simp(diag);

  EXPECT_TRUE(diag.connected(diag.get_inputs()[0], diag.get_outputs()[0]));
  EXPECT_TRUE(diag.connected(diag.get_inputs()[1], diag.get_outputs()[1]));
  EXPECT_TRUE(diag.connected(diag.get_inputs()[2], diag.get_outputs()[2]));
}

TEST_F(SimplifyTest, clifford_3) {
  auto diag = make_empty_diagram(2);
  diag.add_vertex(0);
  diag.add_vertex(0, 0, zx::PyRational(0, 1), zx::VertexType::X);

  diag.add_vertex(0);
  diag.add_vertex(1, 0, zx::PyRational(0, 1), zx::VertexType::X);

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
//   zx::ZXDiagram diag("circuits/ctrl_phase.qasm");

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

  diag.add_vertex(0);                          // 2
  diag.add_vertex(0);                          // 3
  diag.add_vertex(0);                          // 4
  diag.add_vertex(0, 0, zx::PyRational(1, 1)); // 5
  diag.add_vertex(0);                          // 6
  diag.add_vertex(0, 0, zx::PyRational(1, 1)); // 7

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
  EXPECT_TRUE(diag.phase(5).is_zero());
}

TEST_F(SimplifyTest, gadget_simp_2) {
  zx::ZXDiagram diag = make_empty_diagram(1);
  diag.add_vertex(0);                          // 2
  diag.add_vertex(0);                          // 3
  diag.add_vertex(0, 0, zx::PyRational(1, 1)); // 4
  diag.add_vertex(0);                          // 5
  diag.add_vertex(0, 0, zx::PyRational(1, 1)); // 6

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
  EXPECT_TRUE(diag.phase(2).is_zero());
}

TEST_F(SimplifyTest, pivot_gadget_simp) {}
TEST_F(SimplifyTest, full_reduce) {
  zx::ZXDiagram diag("circuits/ctrl_phase.qasm");

  zx::full_reduce(diag);

  EXPECT_TRUE(diag.is_identity());
}

TEST_F(SimplifyTest, full_reduce_2) {
  zx::ZXDiagram diag = make_empty_diagram(2);

  diag.add_vertex(0, 0, zx::PyRational(1, 32), zx::VertexType::X);  // 4
  diag.add_vertex(0, 0, zx::PyRational(0, 1), zx::VertexType::Z);   // 5
  diag.add_vertex(1, 0, zx::PyRational(0, 1), zx::VertexType::X);   // 6
  diag.add_vertex(0, 0, zx::PyRational(0, 1), zx::VertexType::Z);   // 7
  diag.add_vertex(1, 0, zx::PyRational(0, 1), zx::VertexType::X);   // 8
  diag.add_vertex(0, 0, zx::PyRational(-1, 32), zx::VertexType::X); // 9

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
  zx::ZXDiagram diag("circuits/bell_state.qasm");
  auto h = diag;
  diag.invert();
  diag.concat(h);

  zx::full_reduce(diag);

  EXPECT_TRUE(diag.is_identity());
}

TEST_F(SimplifyTest, full_reduce_4) {
  zx::ZXDiagram d0("circuits/C17_204_o0.qasm");
  zx::ZXDiagram d1("circuits/C17_204_o1.qasm");

  d0.invert();
  d0.concat(d1);

  zx::full_reduce(d0);

  EXPECT_TRUE(2 * d0.get_nedges() == d0.get_nvertices());
}

TEST_F(SimplifyTest, full_reduce_5) {
  zx::ZXDiagram d0("circuits/test0.qasm");
  zx::ZXDiagram d1("circuits/test1.qasm");

  d0.invert();
  d0.concat(d1);

  zx::full_reduce(d0);

  EXPECT_EQ(d0.get_nedges(), 3);
  EXPECT_EQ(d0.get_nvertices(), 6);
}

class ExpressionTest : public ::testing::Test {
public:
  // const zx::Variable x_var{0, "x"};
  // const zx::Variable y_var{1, "y"};
  // const zx::Variable z_var{2, "z"};

  zx::Term x{zx::Variable(0, "x")};
  zx::Term y{zx::Variable(1, "y")};
  zx::Term z{zx::Variable(2, "z")};

protected:
  virtual void SetUp() {}
};

TEST_F(ExpressionTest, basic_ops_1) {
  zx::Expression e(x);

  EXPECT_EQ(1, e.num_terms());
  EXPECT_EQ(zx::PyRational(0, 1), e.get_constant());

  e += x; // zx::Term(x);

  EXPECT_EQ(1, e.num_terms());
  EXPECT_EQ(zx::PyRational(0, 1), e.get_constant());
  EXPECT_PRED_FORMAT2(testing::FloatLE, e[0].get_coeff(), 2.0);

  e += y;
  EXPECT_EQ(2, e.num_terms());
  EXPECT_PRED_FORMAT2(testing::FloatLE, e[0].get_coeff(), 2.0);
  EXPECT_PRED_FORMAT2(testing::FloatLE, e[1].get_coeff(), 1.0);
  EXPECT_EQ(e[0].get_var().name, "x");
  EXPECT_EQ(e[1].get_var().name, "y");
}

TEST_F(ExpressionTest, basic_ops_2) {
  zx::Expression e1;
  e1 += x;
  e1 += 10.0 * y;
  e1 += 5.0 * z;
  e1 += zx::PyRational(1, 2);

  zx::Expression e2;
  e2 += -5.0 * x;
  e2 += -10.0 * y;
  e2 += -4.9 * z;
  e2 += zx::PyRational(3, 2);

  auto sum = e1 + e2;

  EXPECT_EQ(2, sum.num_terms());
  EXPECT_PRED_FORMAT2(testing::FloatLE, sum[0].get_coeff(), -4.0);
  EXPECT_PRED_FORMAT2(testing::FloatLE, sum[1].get_coeff(), 0.1);
  EXPECT_EQ(sum[0].get_var().name, "x");
  EXPECT_EQ(sum[1].get_var().name, "z");
  EXPECT_EQ(sum.get_constant(), zx::PyRational(0, 1));
}

class SymbolicTest : public ::testing::Test {
public:
  // const zx::Variable x_var{0, "x"};
  // const zx::Variable y_var{1, "y"};
  // const zx::Variable z_var{2, "z"};

  zx::Term x{zx::Variable(0, "x")};
  zx::Term y{zx::Variable(1, "y")};
  zx::Term z{zx::Variable(2, "z")};

protected:
  virtual void SetUp() {}
};

TEST_F(SymbolicTest, id_simp) {
  int32_t nqubits = 50;

  zx::ZXDiagram diag = make_identity_diagram(nqubits, 100);
  zx::Expression e;
  e += x;
  diag.set_phase(nqubits * 2 + 5, e);
  EXPECT_EQ(e.num_terms(), 1);
  EXPECT_EQ(diag.phase(nqubits * 2 + 5).num_terms(), 1);

  zx::full_reduce(diag);

  EXPECT_EQ(diag.get_nvertices(), 2 * nqubits + 1);
  EXPECT_EQ(diag.get_nedges(), nqubits + 1);
}

TEST_F(SymbolicTest, equivalence) {
  zx::ZXDiagram d1 = make_empty_diagram(3);

  // first circuit
  d1.add_vertex(0, 0); // 6
  d1.add_hadamard_edge(0, 6);

  d1.add_vertex(0, 0);                                      // 7
  d1.add_vertex(1, 0, zx::Expression(), zx::VertexType::X); // 8
  d1.add_edge(7, 8);
  d1.add_edge(6, 7);
  d1.add_edge(1, 8);

  d1.add_vertex(0, 0);                                      // 9
  d1.add_vertex(2, 0, zx::Expression(), zx::VertexType::X); // 10
  d1.add_edge(9, 10);
  d1.add_edge(7, 9);
  d1.add_edge(2, 10);

  d1.add_vertex(0, 0, zx::Expression(x), zx::VertexType::Z); // 11
  d1.add_vertex(1, 0, zx::Expression(y), zx::VertexType::X); // 12
  d1.add_vertex(2, 0, zx::Expression(z), zx::VertexType::X); // 13
  d1.add_edge(9, 11);
  d1.add_edge(8, 12);
  d1.add_edge(10, 13);

  d1.add_vertex(0, 0);                                      // 14
  d1.add_vertex(1, 0, zx::Expression(), zx::VertexType::X); // 15
  d1.add_edge(14, 15);
  d1.add_edge(11, 14);
  d1.add_edge(12, 15);

  d1.add_vertex(0, 0);                                      // 16
  d1.add_vertex(2, 0, zx::Expression(), zx::VertexType::X); // 17
  d1.add_edge(16, 17);
  d1.add_edge(14, 16);
  d1.add_edge(13, 17);

  d1.add_vertex(0, 0); // 18
  d1.add_hadamard_edge(16, 18);

  // second circuit
  d1.add_vertex(0, 0); // 19
  d1.add_hadamard_edge(18, 19);

  d1.add_vertex(0, 0);                                      // 20
  d1.add_vertex(1, 0, zx::Expression(), zx::VertexType::X); // 21
  d1.add_edge(20, 21);
  d1.add_edge(19, 20);
  d1.add_edge(17, 21);

  d1.add_vertex(0, 0);                                      // 22
  d1.add_vertex(1, 0, zx::Expression(), zx::VertexType::X); // 23
  d1.add_edge(22, 23);
  d1.add_edge(20, 22);
  d1.add_edge(15, 23);

  d1.add_vertex(0, 0, -zx::Expression(x), zx::VertexType::Z); // 24
  d1.add_vertex(1, 0, -zx::Expression(y), zx::VertexType::X); // 25
  d1.add_vertex(2, 0, -zx::Expression(z), zx::VertexType::X); // 26
  d1.add_edge(22, 24);
  d1.add_edge(23, 25);
  d1.add_edge(21, 26);

  d1.add_vertex(0, 0);                                      // 27
  d1.add_vertex(1, 0, zx::Expression(), zx::VertexType::X); // 28
  d1.add_edge(24, 27);
  d1.add_edge(26, 28);
  d1.add_edge(28, 27);

  d1.add_vertex(0, 0);                                      // 29
  d1.add_vertex(1, 0, zx::Expression(), zx::VertexType::X); // 30
  d1.add_edge(29, 30);
  d1.add_edge(27, 29);
  d1.add_edge(25, 30);

  d1.add_hadamard_edge(29, 3);
  d1.add_edge(30, 4);
  d1.add_edge(28, 5);

  zx::full_reduce(d1);

  EXPECT_EQ(d1.get_nedges(), 3);
  EXPECT_EQ(d1.get_nvertices(), 6);
  EXPECT_TRUE(d1.is_identity());

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// TEST_F(SimplifyTest, ancillaries) {
//   zx::ZXDiagram d0("circuits/mcx_no_ancilla.qasm");
//   zx::ZXDiagram d1("circuits/mcx_ancilla.qasm");

//   d1.make_ancilla(4);

//   // d1 = d0;
//   d0.invert();
//   d0.concat(d1);

//   zx::full_reduce(d0);
//   for (auto [from, to] : d0.get_edges()) {
//     std::cout << from
//               << (d0.get_edge(from, to).value().type ==
//               zx::EdgeType::Hadamard
//                       ? "- -"
//                       : "---")
//               << to << "\n";
//   }
//   std::cout << ""
//             << "\n";

//   for (int i = 0; i < d0.get_inputs().size(); i++) {
//     std::cout << d0.get_inputs()[i] << "--" << d0.get_outputs()[i] << "\n";
//   }
//   std::cout << ""
//             << "\n";

//   for (auto [v, data] : d0.get_vertices())
//     std::cout << v << " p: " << data.phase <<", q:" << ((int)data.qubit) <<
//     ", r:" << (data.col)<<"\n";
//   std::cout << ""
//             << "\n";
//   for (auto [v, data] : d0.get_vertices()) {
//     std::cout << v << " p:" << data.phase << " boundary "
//               << (data.type == zx::VertexType::Boundary ? "True" : "False")
//               << " type " << (d0.type(v) == zx::VertexType::Z ? "Z" : "X")
//               << "\n";
//   }

//   EXPECT_TRUE(d0.is_identity());
// }
