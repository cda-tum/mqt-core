/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"
#include "Simplify.hpp"
#include "ZXDiagram.hpp"
#include "zx/FunctionalityConstruction.hpp"

#include "gtest/gtest.h"

class ZXDiagramTest: public ::testing::Test {
public:
    std::unique_ptr<qc::QuantumComputation> qc;

protected:
    void TearDown() override {
    }

    void SetUp() override {
        qc = std::make_unique<qc::QuantumComputation>();
    }
};

TEST_F(ZXDiagramTest, parse_qasm) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[2];"
       << "h q[0];"
       << "cx q[0],q[1];"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
    zx::ZXDiagram diag = zx::FunctionalityConstruction::buildFunctionality(qc.get());
    EXPECT_EQ(diag.getNVertices(), 7);
    EXPECT_EQ(diag.getNEdges(), 6);

    auto inputs = diag.getInputs();
    EXPECT_EQ(inputs[0], 0);
    EXPECT_EQ(inputs[1], 1);

    auto outputs = diag.getOutputs();
    EXPECT_EQ(outputs[0], 2);
    EXPECT_EQ(outputs[1], 3);

    EXPECT_EQ(diag.getEdge(0, 4).value().type, zx::EdgeType::Hadamard);
    EXPECT_EQ(diag.getEdge(5, 6).value().type, zx::EdgeType::Simple);
    EXPECT_EQ(diag.getEdge(6, 1).value().type, zx::EdgeType::Simple);
    EXPECT_EQ(diag.getEdge(3, 6).value().type, zx::EdgeType::Simple);
    EXPECT_EQ(diag.getEdge(4, 5).value().type, zx::EdgeType::Simple);
    EXPECT_EQ(diag.getEdge(5, 2).value().type, zx::EdgeType::Simple);

    EXPECT_EQ(diag.getVData(0).value().type, zx::VertexType::Boundary);
    EXPECT_EQ(diag.getVData(1).value().type, zx::VertexType::Boundary);
    EXPECT_EQ(diag.getVData(4).value().type, zx::VertexType::Z);
    EXPECT_EQ(diag.getVData(5).value().type, zx::VertexType::Z);
    EXPECT_EQ(diag.getVData(6).value().type, zx::VertexType::X);
    EXPECT_EQ(diag.getVData(2).value().type, zx::VertexType::Boundary);
    EXPECT_EQ(diag.getVData(3).value().type, zx::VertexType::Boundary);

    for (auto i = 0; i < diag.getNVertices(); i++)
        EXPECT_TRUE(diag.getVData(i).value().phase.isZero());
}

TEST_F(ZXDiagramTest, many_gates) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[3];"
       << "h q[0];"
       << "cx q[0],q[1];"
       << "z q[1];"
       << "x q[2];"
       << "y q[0];"
       << "rx(0.7853981633974483) q[0];"
       << "rz(0.7853981633974483) q[1];"
       << "ry(0.7853981633974483) q[2];"
       << "t q[0];"
       << "s q[2];"
       << "u2(0.7853981633974483, 0.7853981633974483) q[1];"
       << "u3(0.7853981633974483, 0.7853981633974483, 0.7853981633974483) q[2];"
       << "swap q[0],q[1];"
       << "cz q[1],q[2];"
       << "cp(0.7853981633974483) q[0],q[1];"
       << "ccx q[0],q[1],q[2];"
       << "ccz q[1],q[2],q[0];"
       << "ccz q[1],q[2],q[0];"
       << "ccx q[0],q[1],q[2];"
       << "cp(-0.7853981633974483) q[0],q[1];"
       << "cz q[1],q[2];"
       << "cx q[1],q[0];"
       << "cx q[0],q[1];"
       << "cx q[1],q[0];"
       << "u3(-pi/4,-pi/4,-pi/4) q[2];"
       << "u2(-5*pi/4,3*pi/4) q[1];"
       << "sdg q[2];"
       << "tdg q[0];"
       << "ry(-0.7853981633974483) q[2];"
       << "rz(-0.7853981633974483) q[1];"
       << "rx(-0.7853981633974483) q[0];"
       << "y q[0];"
       << "x q[2];"
       << "z q[1];"
       << "cx q[0],q[1];"
       << "h q[0];"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
    zx::ZXDiagram diag = zx::FunctionalityConstruction::buildFunctionality(qc.get());
    zx::fullReduce(diag);
    EXPECT_EQ(diag.getNVertices(), 6);
    EXPECT_EQ(diag.getNEdges(), 3);
    EXPECT_TRUE(diag.isIdentity());

    // EXPECT_EQ(diag.getEdge(0, 4).value().type, zx::EdgeType::Hadamard);
}
