#include "Definitions.hpp"
#include "Expression.hpp"
#include "Rational.hpp"
#include "Simplify.hpp"
#include "ZXDiagram.hpp"

#include <cstdint>
#include <gtest/gtest.h>

using zx::Expression;

class RationalTest: public ::testing::Test {};

TEST_F(RationalTest, normalize) {
    zx::PiRational r(-33, 16);
    EXPECT_EQ(r, zx::PiRational(-1, 16));
}

TEST_F(RationalTest, from_double) {
    zx::PiRational r(-zx::PI / 8);
    EXPECT_EQ(r, zx::PiRational(-1, 8));
}

TEST_F(RationalTest, from_double_2) {
    zx::PiRational r(-3 * zx::PI / 4);
    EXPECT_EQ(r, zx::PiRational(-3, 4));
}

TEST_F(RationalTest, from_double_3) {
    zx::PiRational r(-7 * zx::PI / 8);
    EXPECT_EQ(r, zx::PiRational(-7, 8));
}

TEST_F(RationalTest, from_double_4) {
    zx::PiRational r(-1 * zx::PI / 32);
    EXPECT_EQ(r, zx::PiRational(-1, 32));
}

TEST_F(RationalTest, from_double_5) {
    zx::PiRational r(5000 * zx::PI + zx::PI / 4);
    EXPECT_EQ(r, zx::PiRational(1, 4));
}

TEST_F(RationalTest, from_double_6) {
    zx::PiRational r(-5000 * zx::PI + 5 * zx::PI / 4);
    EXPECT_EQ(r, zx::PiRational(-3, 4));
}

TEST_F(RationalTest, from_double_7) {
    zx::PiRational r(0.1);
    std::cout << r << "\n";
}

TEST_F(RationalTest, add) {
    zx::PiRational r0(1, 8);
    zx::PiRational r1(7, 8);
    auto           r = r0 + r1;

    EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, add_2) {
    zx::PiRational r0(9, 8);
    zx::PiRational r1(7, 8);
    auto           r = r0 + r1;

    EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, sub) {
    zx::PiRational r0(9, 8);
    zx::PiRational r1(-7, 8);
    auto           r = r0 - r1;

    EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, sub_2) {
    zx::PiRational r0(-1, 2);
    zx::PiRational r1(1, 2);
    auto           r = r0 - r1;

    EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, mul) {
    zx::PiRational r0(1, 8);
    zx::PiRational r1(1, 2);
    auto           r = r0 * r1;

    EXPECT_EQ(r, zx::PiRational(1, 16));
}

TEST_F(RationalTest, mul_2) {
    zx::PiRational r0(1, 8);
    zx::PiRational r1(0, 1);
    auto           r = r0 * r1;

    EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, div) {
    zx::PiRational r0(1, 2);
    zx::PiRational r1(1, 2);
    auto           r = r0 / r1;

    EXPECT_EQ(r, 1);
}

class ZXDiagramTest: public ::testing::Test {
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
    virtual void SetUp() {
        diag = zx::ZXDiagram();
        diag.addQubits(2);
        diag.addVertex(0, 0, zx::Expression(), zx::VertexType::Z);
        diag.addEdge(0, 4, zx::EdgeType::Hadamard);
        diag.addVertex(0, 0, zx::Expression(), zx::VertexType::Z);
        diag.addEdge(4, 5);
        diag.addVertex(0, 0, zx::Expression(), zx::VertexType::X);
        diag.addEdge(2, 6);
        diag.addEdge(5, 6);
        diag.addEdge(5, 1);
        diag.addEdge(6, 3);
    }
};

TEST_F(ZXDiagramTest, create_diagram) {
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

    for (std::size_t i = 0; i < diag.getNVertices(); i++)
        EXPECT_TRUE(diag.getVData(6).value().phase.isZero());
}

TEST_F(ZXDiagramTest, deletions) {
    diag.removeVertex(5);
    EXPECT_EQ(diag.getNVertices(), 6);
    EXPECT_EQ(diag.getNEdges(), 3);
    EXPECT_FALSE(diag.getVData(5).has_value());

    diag.remove_edge(0, 4);
    EXPECT_EQ(diag.getNVertices(), 6);
    EXPECT_EQ(diag.getNEdges(), 2);
}

TEST_F(ZXDiagramTest, graph_like) {
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

    for (std::size_t i = 0; i < diag.getNVertices(); i++)
        EXPECT_TRUE(diag.getVData(i).value().phase.isZero());
}

// TEST_F(ZXDiagramTest, concat) {
//     auto copy = diag;
//     diag.concat(copy);

//     ASSERT_EQ(diag.getNEdges(), 10);
//     ASSERT_EQ(diag.getNVertices(), 10);

//     EXPECT_EQ(diag.getEdge(0, 4).value().type, zx::EdgeType::Hadamard);
//     EXPECT_EQ(diag.getEdge(5, 6).value().type, zx::EdgeType::Simple);
//     EXPECT_EQ(diag.getEdge(2, 6).value().type, zx::EdgeType::Simple);
//     EXPECT_EQ(diag.getEdge(3, 6).value().type, zx::EdgeType::Simple);
//     EXPECT_EQ(diag.getEdge(5, 9).value().type, zx::EdgeType::Hadamard);
//     // EXPECT_EQ(diag.getEdge(4, 9).value().type, zx::EdgeType::Simple);
//     // EXPECT_EQ(diag.getEdge(7, 8).value().type, zx::EdgeType::Simple);
//     // EXPECT_EQ(diag.getEdge(8, 10).value().type, zx::EdgeType::Simple);
//     // EXPECT_EQ(diag.getEdge(9, 11).value().type, zx::EdgeType::Simple);

//     // EXPECT_EQ(diag.getVData(0).value().type, zx::VertexType::Boundary);
//     // EXPECT_EQ(diag.getVData(1).value().type, zx::VertexType::Boundary);
//     // EXPECT_EQ(diag.getVData(2).value().type, zx::VertexType::Z);
//     // EXPECT_EQ(diag.getVData(3).value().type, zx::VertexType::Z);
//     // EXPECT_EQ(diag.getVData(4).value().type, zx::VertexType::X);
//     // EXPECT_EQ(diag.getVData(7).value().type, zx::VertexType::Z);
//     // EXPECT_EQ(diag.getVData(8).value().type, zx::VertexType::Z);
//     // EXPECT_EQ(diag.getVData(9).value().type, zx::VertexType::X);
//     // EXPECT_EQ(diag.getVData(10).value().type, zx::VertexType::Boundary);
//     // EXPECT_EQ(diag.getVData(11).value().type, zx::VertexType::Boundary);

//     EXPECT_TRUE(diag.isDeleted(1));
//     EXPECT_TRUE(diag.isDeleted(3));
// }

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

class SimplifyTest: public ::testing::Test {
public:
protected:
    virtual void SetUp() {}
};

zx::ZXDiagram make_identity_diagram(int32_t nqubits,
                                    int32_t spiders_per_qubit) {
    zx::ZXDiagram           diag(nqubits);
    std::vector<zx::Vertex> rightmost_vertices = diag.getInputs();

    for (auto i = 0; i < nqubits; i++)
        diag.remove_edge(i, i + nqubits);

    // add identity spiders
    for (auto qubit = 0; qubit < nqubits; qubit++) {
        for (auto j = 0; j < spiders_per_qubit; j++) {
            zx::Vertex v = diag.addVertex(qubit);
            diag.addEdge(rightmost_vertices[qubit], v);
            rightmost_vertices[qubit] = v;
        }
    }

    for (auto qubit = 0; qubit < nqubits; qubit++) {
        diag.addEdge(rightmost_vertices[qubit], qubit + nqubits);
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

TEST_F(SimplifyTest, idSimp) {
    int32_t       nqubits = 3;
    int32_t       spiders = 100;
    zx::ZXDiagram diag    = make_identity_diagram(nqubits, spiders);

    int32_t removed = zx::idSimp(diag);

    EXPECT_EQ(removed, nqubits * spiders);
    EXPECT_EQ(diag.getNVertices(), nqubits * 2);
    EXPECT_EQ(diag.getNEdges(), nqubits);
}

TEST_F(SimplifyTest, idSimp_2) {
    int32_t       nqubits = 2;
    int32_t       spiders = 100;
    zx::ZXDiagram diag    = make_identity_diagram(nqubits, spiders);

    diag.addEdge(50, 150); // make vertices 50 and 150 non-removable

    int32_t removed = zx::idSimp(diag);
    EXPECT_EQ(removed, nqubits * 100 - 2);
    EXPECT_EQ(diag.getNVertices(), nqubits * 2 + 2);
    EXPECT_EQ(diag.getNEdges(), 5);
}

TEST_F(SimplifyTest, spider_fusion) {
    int32_t       nqubits  = 1;
    int32_t       nspiders = 100;
    zx::ZXDiagram diag     = make_identity_diagram(nqubits, nspiders);

    for (zx::Vertex v = 2; v < diag.getNVertices(); v++)
        diag.addPhase(v, zx::Expression(zx::PiRational(1, 1)));

    int32_t removed = zx::spiderSimp(diag);

    EXPECT_EQ(removed, nspiders - 1);
    EXPECT_EQ(3, diag.getNVertices());
    EXPECT_EQ(2, diag.getNEdges());
    EXPECT_TRUE(diag.phase(2).isZero());
}

TEST_F(SimplifyTest, spider_fusion_2) {
    int32_t       nqubits  = 2;
    int32_t       nspiders = 5;
    zx::ZXDiagram diag     = make_identity_diagram(nqubits, nspiders);

    diag.addEdge(6, 11);

    int32_t removed = zx::spiderSimp(diag);

    EXPECT_EQ(removed, 9);
    EXPECT_EQ(diag.getNVertices(), 5);
    EXPECT_EQ(diag.getNEdges(), 4);

    zx::Vertex interior = diag.incidentEdges(0)[0].to;
    for (zx::Vertex v: diag.getInputs()) {
        EXPECT_TRUE(diag.connected(v, interior));
    }
    for (zx::Vertex v: diag.getOutputs()) {
        EXPECT_TRUE(diag.connected(v, interior));
    }
}

TEST_F(SimplifyTest, spider_fusion_parallel_edges) {
    int32_t       nqubits  = 1;
    int32_t       nspiders = 3;
    zx::ZXDiagram diag     = make_identity_diagram(nqubits, nspiders);
    diag.addEdge(2, 4);
    diag.setType(4, zx::VertexType::X);

    int32_t removed = zx::spiderSimp(diag);

    EXPECT_EQ(removed, 1);
    EXPECT_EQ(diag.getNEdges(), 2);
    EXPECT_EQ(diag.incidentEdges(1).size(), 1);
}

TEST_F(SimplifyTest, localComp) {
    zx::ZXDiagram diag(2);
    diag.remove_edge(0, 2);
    diag.remove_edge(1, 3);

    diag.addVertex(0, 0, zx::PiRational(1, 2), zx::VertexType::Z); // 4
    diag.addVertex(0, 0, zx::PiRational(0, 1), zx::VertexType::Z); // 5
    diag.addVertex(0, 0, zx::PiRational(0, 1), zx::VertexType::Z); // 6
    diag.addVertex(0, 0, zx::PiRational(0, 1), zx::VertexType::Z); // 7
    diag.addVertex(0, 0, zx::PiRational(0, 1), zx::VertexType::Z); // 8

    diag.addEdge(4, 5, zx::EdgeType::Hadamard);
    diag.addEdge(4, 6, zx::EdgeType::Hadamard);
    diag.addEdge(4, 7, zx::EdgeType::Hadamard);
    diag.addEdge(4, 8, zx::EdgeType::Hadamard);

    diag.addEdge(0, 5);
    diag.addEdge(1, 6);
    diag.addEdge(2, 7);
    diag.addEdge(3, 8);

    int32_t removed = zx::localCompSimp(diag);

    EXPECT_EQ(removed, 1);

    for (zx::Vertex v = 5; v <= 8; v++) {
        EXPECT_TRUE(diag.phase(v) == zx::Expression(zx::PiRational(-1, 2)));
        for (zx::Vertex w = 5; w <= 8; w++) {
            if (w != v) {
                ASSERT_TRUE(diag.connected(v, w));
                EXPECT_EQ(diag.getEdge(v, w).value().type, zx::EdgeType::Hadamard);
            }
        }
    }
}

TEST_F(SimplifyTest, pivotPauli) {
    zx::ZXDiagram diag = make_identity_diagram(2, 0);

    // remove edges between input and outputs
    diag.remove_edge(0, 2);
    diag.remove_edge(1, 3);

    diag.addVertex(0, 0, zx::PiRational(1, 1), zx::VertexType::Z); // 4
    diag.addVertex(0, 0, zx::PiRational(0, 1), zx::VertexType::Z); // 5
    diag.addVertex(0, 0, zx::PiRational(0, 1), zx::VertexType::Z); // 6
    diag.addVertex(1, 0, zx::PiRational(0, 1), zx::VertexType::Z); // 7
    diag.addVertex(0, 0, zx::PiRational(0, 1), zx::VertexType::Z); // 8
    diag.addVertex(1, 0, zx::PiRational(0, 1), zx::VertexType::Z); // 9
    diag.addVertex(1, 0, zx::PiRational(0, 1), zx::VertexType::Z); // 10

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

    int32_t removed = zx::pivotPauliSimp(diag);

    EXPECT_EQ(removed, 1);
    EXPECT_EQ(diag.getNEdges(), 12);
    EXPECT_EQ(diag.getNVertices(), 9);
    EXPECT_TRUE(diag.phase(8) == zx::Expression(zx::PiRational(1, 1)));
    EXPECT_TRUE(diag.phase(9) == zx::PiRational(1, 1));
    EXPECT_TRUE(diag.phase(10) == zx::PiRational(0, 1));
    EXPECT_TRUE(diag.phase(6) == zx::PiRational(0, 1));
    EXPECT_TRUE(diag.phase(7) == zx::PiRational(0, 1));
}

TEST_F(SimplifyTest, interior_clifford) {
    int32_t       nqubits       = 100;
    int32_t       qubit_spiders = 100;
    zx::ZXDiagram diag          = make_identity_diagram(nqubits, qubit_spiders);

    zx::interiorCliffordSimp(diag);

    EXPECT_EQ(diag.getNVertices(), nqubits * 2);
    EXPECT_EQ(diag.getNEdges(), nqubits);
    for (auto v = 0; v < nqubits; v++) {
        EXPECT_TRUE(diag.connected(diag.getInputs()[v], diag.getOutputs()[v]));
    }
}

TEST_F(SimplifyTest, interior_clifford_2) {
    zx::ZXDiagram diag(1);
    diag.remove_edge(0, 1);

    diag.addVertex(0, 0, zx::PiRational(-1, 2), zx::VertexType::X); // 2
    diag.addVertex(0, 0, zx::PiRational(1, 2), zx::VertexType::Z);  // 3
    diag.addVertex(0, 0, zx::PiRational(-1, 2), zx::VertexType::X); // 4

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
}

TEST_F(SimplifyTest, non_pauli_pivot) {
    zx::ZXDiagram diag(1);
    diag.remove_edge(0, 1);

    diag.addVertex(0, 0, zx::PiRational(1, 4)); // 2
    diag.addVertex(0);                          // 3
    diag.addVertex(0);                          // 4

    diag.addEdge(0, 2);
    diag.addEdge(2, 3, zx::EdgeType::Hadamard);
    diag.addEdge(3, 4, zx::EdgeType::Hadamard);
    diag.addEdge(4, 1);

    diag.toGraphlike();
    auto res = zx::pivotSimp(diag);

    EXPECT_GT(res, 0);
    ASSERT_EQ(diag.getNEdges(), 5);
    ASSERT_EQ(diag.getNVertices(), 6);

    EXPECT_TRUE(diag.connected(0, 7));
    EXPECT_TRUE(diag.connected(7, 4));
    EXPECT_TRUE(diag.connected(1, 4));
    EXPECT_TRUE(diag.connected(6, 4));
    EXPECT_TRUE(diag.connected(5, 6));
}

// TEST_F(SimplifyTest, clifford) {
//     zx::ZXDiagram diag("circuits/clifford_identity_simple.qasm");
//     diag.toGraphlike();
//     zx::cliffordSimp(diag);

//     EXPECT_TRUE(diag.connected(diag.getInputs()[0], diag.getOutputs()[0]));
// }

// TEST_F(SimplifyTest, clifford_2) {
//     zx::ZXDiagram diag("circuits/ghz_identity.qasm");

//     diag.toGraphlike();

//     zx::cliffordSimp(diag);

//     EXPECT_TRUE(diag.connected(diag.getInputs()[0], diag.getOutputs()[0]));
//     EXPECT_TRUE(diag.connected(diag.getInputs()[1], diag.getOutputs()[1]));
//     EXPECT_TRUE(diag.connected(diag.getInputs()[2], diag.getOutputs()[2]));
// }

// TEST_F(SimplifyTest, clifford_3) {
//     auto diag = make_empty_diagram(2);
//     diag.addVertex(0);
//     diag.addVertex(0, 0, zx::PiRational(0, 1), zx::VertexType::X);

//     diag.addVertex(0);
//     diag.addVertex(1, 0, zx::PiRational(0, 1), zx::VertexType::X);

//     diag.addEdge(0, 4);
//     diag.addEdge(1, 5);
//     diag.addEdge(4, 5);
//     diag.addEdge(4, 6);
//     diag.addEdge(5, 7);
//     diag.addEdge(6, 7);
//     diag.addEdge(6, 2);
//     diag.addEdge(7, 3);

//     //    zx::spiderSimp(diag);
//     zx::cliffordSimp(diag);
//     EXPECT_TRUE(diag.connected(diag.getInputs()[0], diag.getOutputs()[0]));
//     EXPECT_TRUE(diag.connected(diag.getInputs()[1], diag.getOutputs()[1]));
// }

// // TEST_F(SimplifyTest, non_clifford) {
// //   zx::ZXDiagram diag("circuits/ctrl_phase.qasm");

// //   for (auto [to, from] : diag.getEdges()) {
// //     std::cout << to << "-" << from << "\n";
// //   }
// //   std::cout << ""
// //             << "\n";

// //   diag.toGraphlike();
// //   zx::cliffordSimp(diag);

// //  for (auto [to, from] : diag.getEdges()) {
// //    std::cout << to << "-" << from << "\n";
// //   }
// // }

TEST_F(SimplifyTest, gadgetSimp) {
    zx::ZXDiagram diag = make_empty_diagram(1);

    diag.addVertex(0);                          // 2
    diag.addVertex(0);                          // 3
    diag.addVertex(0);                          // 4
    diag.addVertex(0, 0, zx::PiRational(1, 1)); // 5
    diag.addVertex(0);                          // 6
    diag.addVertex(0, 0, zx::PiRational(1, 1)); // 7

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
}

TEST_F(SimplifyTest, gadgetSimp_2) {
    zx::ZXDiagram diag = make_empty_diagram(1);
    diag.addVertex(0);                          // 2
    diag.addVertex(0);                          // 3
    diag.addVertex(0, 0, zx::PiRational(1, 1)); // 4
    diag.addVertex(0);                          // 5
    diag.addVertex(0, 0, zx::PiRational(1, 1)); // 6

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
}

// TEST_F(SimplifyTest, pivotgadgetSimp) {}
// TEST_F(SimplifyTest, fullReduce) {
//     zx::ZXDiagram diag("circuits/ctrl_phase.qasm");

//     zx::fullReduce(diag);

//     EXPECT_TRUE(diag.isIdentity());
// }

TEST_F(SimplifyTest, fullReduce_2) {
    zx::ZXDiagram diag = make_empty_diagram(2);

    diag.addVertex(0, 0, zx::PiRational(1, 32), zx::VertexType::X);  // 4
    diag.addVertex(0, 0, zx::PiRational(0, 1), zx::VertexType::Z);   // 5
    diag.addVertex(1, 0, zx::PiRational(0, 1), zx::VertexType::X);   // 6
    diag.addVertex(0, 0, zx::PiRational(0, 1), zx::VertexType::Z);   // 7
    diag.addVertex(1, 0, zx::PiRational(0, 1), zx::VertexType::X);   // 8
    diag.addVertex(0, 0, zx::PiRational(-1, 32), zx::VertexType::X); // 9

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

// TEST_F(SimplifyTest, fullReduce_3) {
//     zx::ZXDiagram diag("circuits/bell_state.qasm");
//     auto          h = diag;
//     diag.invert();
//     diag.concat(h);

//     zx::fullReduce(diag);

//     EXPECT_TRUE(diag.isIdentity());
// }

// TEST_F(SimplifyTest, fullReduce_4) {
//     zx::ZXDiagram d0("circuits/C17_204_o0.qasm");
//     zx::ZXDiagram d1("circuits/C17_204_o1.qasm");

//     d0.invert();
//     d0.concat(d1);

//     zx::fullReduce(d0);

//     EXPECT_TRUE(2 * d0.getNEdges() == d0.getNVertices());
// }

// TEST_F(SimplifyTest, fullReduce_5) {
//     zx::ZXDiagram d0("circuits/test0.qasm");
//     zx::ZXDiagram d1("circuits/test1.qasm");

//     d0.invert();
//     d0.concat(d1);

//     zx::fullReduce(d0);

//     EXPECT_EQ(d0.getNEdges(), 3);
//     EXPECT_EQ(d0.getNVertices(), 6);
// }

class ExpressionTest: public ::testing::Test {
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

    EXPECT_EQ(1, e.numTerms());
    EXPECT_EQ(zx::PiRational(0, 1), e.getConst());

    e += x; // zx::Term(x);

    EXPECT_EQ(1, e.numTerms());
    EXPECT_EQ(zx::PiRational(0, 1), e.getConst());
    EXPECT_PRED_FORMAT2(testing::FloatLE, e[0].getCoeff(), 2.0);

    e += y;
    EXPECT_EQ(2, e.numTerms());
    EXPECT_PRED_FORMAT2(testing::FloatLE, e[0].getCoeff(), 2.0);
    EXPECT_PRED_FORMAT2(testing::FloatLE, e[1].getCoeff(), 1.0);
    EXPECT_EQ(e[0].getVar().name, "x");
    EXPECT_EQ(e[1].getVar().name, "y");
}

TEST_F(ExpressionTest, basic_ops_2) {
    zx::Expression e1;
    e1 += x;
    e1 += 10.0 * y;
    e1 += 5.0 * z;
    e1 += zx::PiRational(1, 2);

    zx::Expression e2;
    e2 += -5.0 * x;
    e2 += -10.0 * y;
    e2 += -4.9 * z;
    e2 += zx::PiRational(3, 2);

    auto sum = e1 + e2;

    EXPECT_EQ(2, sum.numTerms());
    EXPECT_PRED_FORMAT2(testing::FloatLE, sum[0].getCoeff(), -4.0);
    EXPECT_PRED_FORMAT2(testing::FloatLE, sum[1].getCoeff(), 0.1);
    EXPECT_EQ(sum[0].getVar().name, "x");
    EXPECT_EQ(sum[1].getVar().name, "z");
    EXPECT_EQ(sum.getConst(), zx::PiRational(0, 1));
}

class SymbolicTest: public ::testing::Test {
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

TEST_F(SymbolicTest, idSimp) {
    int32_t nqubits = 50;

    zx::ZXDiagram  diag = make_identity_diagram(nqubits, 100);
    zx::Expression e;
    e += x;
    diag.setPhase(nqubits * 2 + 5, e);
    EXPECT_EQ(e.numTerms(), 1);
    EXPECT_EQ(diag.phase(nqubits * 2 + 5).numTerms(), 1);

    zx::fullReduce(diag);

    EXPECT_EQ(diag.getNVertices(), 2 * nqubits + 1);
    EXPECT_EQ(diag.getNEdges(), nqubits + 1);
}

TEST_F(SymbolicTest, equivalence) {
    zx::ZXDiagram d1 = make_empty_diagram(3);

    // first circuit
    d1.addVertex(0, 0); // 6
    d1.addHadamardEdge(0, 6);

    d1.addVertex(0, 0);                                      // 7
    d1.addVertex(1, 0, zx::Expression(), zx::VertexType::X); // 8
    d1.addEdge(7, 8);
    d1.addEdge(6, 7);
    d1.addEdge(1, 8);

    d1.addVertex(0, 0);                                      // 9
    d1.addVertex(2, 0, zx::Expression(), zx::VertexType::X); // 10
    d1.addEdge(9, 10);
    d1.addEdge(7, 9);
    d1.addEdge(2, 10);

    d1.addVertex(0, 0, zx::Expression(x), zx::VertexType::Z); // 11
    d1.addVertex(1, 0, zx::Expression(y), zx::VertexType::X); // 12
    d1.addVertex(2, 0, zx::Expression(z), zx::VertexType::X); // 13
    d1.addEdge(9, 11);
    d1.addEdge(8, 12);
    d1.addEdge(10, 13);

    d1.addVertex(0, 0);                                      // 14
    d1.addVertex(1, 0, zx::Expression(), zx::VertexType::X); // 15
    d1.addEdge(14, 15);
    d1.addEdge(11, 14);
    d1.addEdge(12, 15);

    d1.addVertex(0, 0);                                      // 16
    d1.addVertex(2, 0, zx::Expression(), zx::VertexType::X); // 17
    d1.addEdge(16, 17);
    d1.addEdge(14, 16);
    d1.addEdge(13, 17);

    d1.addVertex(0, 0); // 18
    d1.addHadamardEdge(16, 18);

    // second circuit
    d1.addVertex(0, 0); // 19
    d1.addHadamardEdge(18, 19);

    d1.addVertex(0, 0);                                      // 20
    d1.addVertex(1, 0, zx::Expression(), zx::VertexType::X); // 21
    d1.addEdge(20, 21);
    d1.addEdge(19, 20);
    d1.addEdge(17, 21);

    d1.addVertex(0, 0);                                      // 22
    d1.addVertex(1, 0, zx::Expression(), zx::VertexType::X); // 23
    d1.addEdge(22, 23);
    d1.addEdge(20, 22);
    d1.addEdge(15, 23);

    d1.addVertex(0, 0, -zx::Expression(x), zx::VertexType::Z); // 24
    d1.addVertex(1, 0, -zx::Expression(y), zx::VertexType::X); // 25
    d1.addVertex(2, 0, -zx::Expression(z), zx::VertexType::X); // 26
    d1.addEdge(22, 24);
    d1.addEdge(23, 25);
    d1.addEdge(21, 26);

    d1.addVertex(0, 0);                                      // 27
    d1.addVertex(1, 0, zx::Expression(), zx::VertexType::X); // 28
    d1.addEdge(24, 27);
    d1.addEdge(26, 28);
    d1.addEdge(28, 27);

    d1.addVertex(0, 0);                                      // 29
    d1.addVertex(1, 0, zx::Expression(), zx::VertexType::X); // 30
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

// int main(int argc, char** argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }

// TEST_F(SimplifyTest, ancillaries) {
//   zx::ZXDiagram d0("circuits/mcx_no_ancilla.qasm");
//   zx::ZXDiagram d1("circuits/mcx_ancilla.qasm");

//   d1.makeAncilla(4);

//   // d1 = d0;
//   d0.invert();
//   d0.concat(d1);

//   zx::fullReduce(d0);
//   for (auto [from, to] : d0.getEdges()) {
//     std::cout << from
//               << (d0.getEdge(from, to).value().type ==
//               zx::EdgeType::Hadamard
//                       ? "- -"
//                       : "---")
//               << to << "\n";
//   }
//   std::cout << ""
//             << "\n";

//   for (int i = 0; i < d0.getInputs().size(); i++) {
//     std::cout << d0.getInputs()[i] << "--" << d0.getOutputs()[i] << "\n";
//   }
//   std::cout << ""
//             << "\n";

//   for (auto [v, data] : d0.getVertices())
//     std::cout << v << " p: " << data.phase <<", q:" << ((int)data.qubit) <<
//     ", r:" << (data.col)<<"\n";
//   std::cout << ""
//             << "\n";
//   for (auto [v, data] : d0.getVertices()) {
//     std::cout << v << " p:" << data.phase << " boundary "
//               << (data.type == zx::VertexType::Boundary ? "True" : "False")
//               << " type " << (d0.type(v) == zx::VertexType::Z ? "Z" : "X")
//               << "\n";
//   }

//   EXPECT_TRUE(d0.isIdentity());
// }
