/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "Simplify.hpp"
#include "ZXDiagram.hpp"
#include "dd/Control.hpp"
#include "zx/FunctionalityConstruction.hpp"

#include "gtest/gtest.h"
#include <cstddef>
#include <iostream>
#include <sstream>

using namespace dd::literals;

class ZXDiagramTest: public ::testing::Test {
public:
    qc::QuantumComputation qc;
};

TEST_F(ZXDiagramTest, parse_qasm) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[2];"
       << "h q[0];"
       << "cx q[0],q[1];"
       << std::endl;
    qc.import(ss, qc::OpenQASM);
    EXPECT_TRUE(zx::FunctionalityConstruction::transformableToZX(&qc));
    zx::ZXDiagram diag = zx::FunctionalityConstruction::buildFunctionality(&qc);
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

    for (std::size_t i = 0; i < diag.getNVertices(); i++)
        EXPECT_TRUE(diag.getVData(i).value().phase.isZero());
}

TEST_F(ZXDiagramTest, complex_circuit) {
    std::stringstream ss{};
    ss << "// i 1 0 2\n"
       << "// o 0 1 2\n"
       << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[3];"
       << "h q[0];"
       << "cx q[0],q[1];"
       << "z q[1];"
       << "x q[2];"
       << "y q[0];"
       << "rx(pi/4) q[0];"
       << "rz(0.1) q[1];"
       << "p(0.1) q[1];"
       << "ry(pi/4) q[2];"
       << "t q[0];"
       << "s q[2];"
       << "u2(pi/4, pi/4) q[1];"
       << "u3(pi/4, pi/4, pi/4) q[2];"
       << "barrier q[0],q[1],q[2];"
       << "swap q[0],q[1];"
       << "cz q[1],q[2];"
       << "cp(pi/4) q[0],q[1];"
       << "ccx q[0],q[1],q[2];"
       << "ccz q[1],q[2],q[0];"
       << "cp(pi/2) q[0], q[1];"
       << "cp(pi/4) q[0], q[1];"
       << "cp(-pi/4) q[0], q[1];"
       << "cp(-pi/2) q[0], q[1];"
       << "ccz q[1],q[2],q[0];"
       << "ccx q[0],q[1],q[2];"
       << "cp(-pi/4) q[0],q[1];"
       << "cz q[1],q[2];"
       << "cx q[1],q[0];"
       << "cx q[0],q[1];"
       << "cx q[1],q[0];"
       << "u3(-pi/4,-pi/4,-pi/4) q[2];"
       << "u2(-5*pi/4,3*pi/4) q[1];"
       << "sdg q[2];"
       << "tdg q[0];"
       << "ry(-pi/4) q[2];"
       << "p(-0.1) q[1];"
       << "rz(-0.1) q[1];"
       << "rx(-pi/4) q[0];"
       << "y q[0];"
       << "x q[2];"
       << "z q[1];"
       << "cx q[0],q[1];"
       << "h q[0];"
       << std::endl;
    qc.import(ss, qc::OpenQASM);

    EXPECT_TRUE(zx::FunctionalityConstruction::transformableToZX(&qc));
    zx::ZXDiagram diag = zx::FunctionalityConstruction::buildFunctionality(&qc);
    zx::fullReduce(diag);
    EXPECT_EQ(diag.getNVertices(), 6);
    EXPECT_EQ(diag.getNEdges(), 3);
    EXPECT_TRUE(diag.connected(diag.getInput(0), diag.getOutput(0)));
    EXPECT_TRUE(diag.connected(diag.getInput(1), diag.getOutput(1)));
    EXPECT_TRUE(diag.connected(diag.getInput(2), diag.getOutput(2)));
}

TEST_F(ZXDiagramTest, Phase) {
    qc = qc::QuantumComputation(2);
    qc.phase(0, zx::PI / 4);
    qc.phase(0, 1_pc, zx::PI / 4);
    qc.phase(0, 1_pc, -zx::PI / 4);
    qc.phase(0, -zx::PI / 4);

    EXPECT_TRUE(zx::FunctionalityConstruction::transformableToZX(&qc));
    zx::ZXDiagram diag = zx::FunctionalityConstruction::buildFunctionality(&qc);
    zx::fullReduce(diag);

    EXPECT_TRUE(diag.isIdentity());
}

TEST_F(ZXDiagramTest, Compound) {
    std::stringstream ss;
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "gate toff q0,q1,q2 {h q2;cx q1,q2;p(-pi/4) q2;cx q0,q2;p(pi/4) q2;cx q1,q2;p(pi/4) q1;p(-pi/4) q2;cx q0,q2;cx q0,q1;p(pi/4) q0;p(-pi/4) q1;cx q0,q1;p(pi/4) q2;h q2;}"
       << "qreg q[3];"
       << "toff q[0],q[1],q[2];"
       << "ccx q[0],q[1],q[2];"
       << std::endl;

    qc.import(ss, qc::OpenQASM);
    EXPECT_TRUE(zx::FunctionalityConstruction::transformableToZX(&qc));
    zx::ZXDiagram diag = zx::FunctionalityConstruction::buildFunctionality(&qc);
    zx::fullReduce(diag);

    EXPECT_TRUE(diag.isIdentity());
}

TEST_F(ZXDiagramTest, UnsupportedMultiControl) {
    qc = qc::QuantumComputation(4);
    qc.x(0, {1_pc,
             2_pc,
             3_pc});
    EXPECT_FALSE(zx::FunctionalityConstruction::transformableToZX(&qc));
    EXPECT_THROW(zx::ZXDiagram diag = zx::FunctionalityConstruction::buildFunctionality(&qc), zx::ZXException);
}

TEST_F(ZXDiagramTest, UnsupportedControl) {
    qc = qc::QuantumComputation(2);
    qc.y(0, 1_pc);
    EXPECT_FALSE(zx::FunctionalityConstruction::transformableToZX(&qc));
    EXPECT_THROW(zx::ZXDiagram diag = zx::FunctionalityConstruction::buildFunctionality(&qc), zx::ZXException);
}

TEST_F(ZXDiagramTest, UnsupportedControl2) {
    qc = qc::QuantumComputation(3);
    qc.y(0, {1_pc, 2_pc});
    EXPECT_FALSE(zx::FunctionalityConstruction::transformableToZX(&qc));
    EXPECT_THROW(zx::ZXDiagram diag =
                         zx::FunctionalityConstruction::buildFunctionality(&qc),
                 zx::ZXException);
}

TEST_F(ZXDiagramTest, InitialLayout) {
    qc = qc::QuantumComputation(2);
    qc::Permutation layout{};
    layout[0]        = 1;
    layout[1]        = 0;
    qc.initialLayout = layout;
    qc.x(0);
    qc.z(1);

    auto qcPrime = qc::QuantumComputation(2);
    qcPrime.x(1);
    qcPrime.z(0);

    auto d      = zx::FunctionalityConstruction::buildFunctionality(&qc);
    auto dPrime = zx::FunctionalityConstruction::buildFunctionality(&qcPrime);

    d.concat(dPrime);

    zx::fullReduce(d);
    EXPECT_TRUE(d.isIdentity());
}

TEST_F(ZXDiagramTest, RZ) {
    qc = qc::QuantumComputation(1);
    qc.rz(0, zx::PI / 8);

    auto qcPrime = qc::QuantumComputation(1);
    qcPrime.phase(0, zx::PI / 8);

    auto d      = zx::FunctionalityConstruction::buildFunctionality(&qc);
    auto dPrime = zx::FunctionalityConstruction::buildFunctionality(&qcPrime);

    d.concat(dPrime.invert());

    zx::fullReduce(d);
    EXPECT_FALSE(d.isIdentity());
    EXPECT_FALSE(d.globalPhaseIsZero());
    EXPECT_TRUE(d.connected(d.getInput(0), d.getOutput(0)));
}
