/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "CircuitOptimizer.hpp"
#include "QuantumComputation.hpp"
#include "algorithms/RandomCliffordCircuit.hpp"
#include "dd/Control.hpp"
#include "dd/FunctionalityConstruction.hpp"

#include "gtest/gtest.h"
#include <iostream>
#include <random>

using namespace qc;
using namespace dd;

class QFRFunctionality: public testing::TestWithParam<dd::QubitCount> {
protected:
    void TearDown() override {
    }

    void SetUp() override {
        dd = std::make_unique<dd::Package<>>(5);

        std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> random_data{};
        std::random_device                                                    rd;
        std::generate(begin(random_data), end(random_data), [&]() { return rd(); });
        std::seed_seq seeds(begin(random_data), end(random_data));
        mt.seed(seeds);
        dist = std::uniform_real_distribution<dd::fp>(0.0, 2 * dd::PI);
    }

    std::unique_ptr<dd::Package<>>         dd;
    std::mt19937_64                        mt;
    std::uniform_real_distribution<dd::fp> dist;
};

TEST_F(QFRFunctionality, fuse_cx_to_swap) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0_pc, 1, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 1_pc, 0, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 0_pc, 1, qc::X);
    CircuitOptimizer::swapReconstruction(qc);
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>((qc.begin()->get()));
        EXPECT_EQ(op->getType(), SWAP);
        EXPECT_EQ(op->getTargets().at(0), 0);
        EXPECT_EQ(op->getTargets().at(1), 1);
    });
}

TEST_F(QFRFunctionality, replace_cx_to_swap_at_end) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0_pc, 1, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 1_pc, 0, qc::X);
    CircuitOptimizer::swapReconstruction(qc);
    auto it = qc.begin();
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), SWAP);
        EXPECT_EQ(op->getTargets().at(0), 0);
        EXPECT_EQ(op->getTargets().at(1), 1);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::X);
        EXPECT_EQ(op->getControls().begin()->qubit, 0);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
}

TEST_F(QFRFunctionality, replace_cx_to_swap) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0_pc, 1, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 1_pc, 0, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::H);
    CircuitOptimizer::swapReconstruction(qc);
    auto it = qc.begin();
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), SWAP);
        EXPECT_EQ(op->getTargets().at(0), 0);
        EXPECT_EQ(op->getTargets().at(1), 1);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::X);
        EXPECT_EQ(op->getControls().begin()->qubit, 0);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
}

TEST_F(QFRFunctionality, remove_trailing_idle_qubits) {
    QubitCount         nqubits = 4;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 2, qc::X);
    std::cout << qc;
    qc::QuantumComputation::printPermutation(qc.outputPermutation);
    qc.printRegisters();

    qc.outputPermutation.erase(1);
    qc.outputPermutation.erase(3);

    qc.stripIdleQubits();
    EXPECT_EQ(qc.getNqubits(), 2);
    std::cout << qc;
    qc::QuantumComputation::printPermutation(qc.outputPermutation);
    qc.printRegisters();

    qc.pop_back();
    qc.outputPermutation.erase(2);
    std::cout << qc;
    qc::QuantumComputation::printPermutation(qc.outputPermutation);
    qc.printRegisters();

    qc.stripIdleQubits();
    EXPECT_EQ(qc.getNqubits(), 1);
}

TEST_F(QFRFunctionality, ancillary_qubit_at_end) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::X);
    qc.addAncillaryRegister(1);
    EXPECT_EQ(qc.getNancillae(), 1);
    EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
    EXPECT_EQ(qc.getNqubits(), 3);
    qc.emplace_back<StandardOperation>(nqubits, 2, qc::X);
    auto e = dd->createInitialMatrix(qc.getNqubits(), qc.ancillary);
    EXPECT_EQ(e.p->e[0], dd->makeIdent(nqubits));
    EXPECT_EQ(e.p->e[1], MatrixDD::zero);
    EXPECT_EQ(e.p->e[2], MatrixDD::zero);
    EXPECT_EQ(e.p->e[3], MatrixDD::zero);
    auto f = dd->makeIdent(nqubits + 1);
    dd->incRef(f);
    f = dd->reduceAncillae(f, qc.ancillary);
    f = dd->reduceGarbage(f, qc.garbage);
    EXPECT_EQ(e, f);
    qc.printRegisters();
    auto p = qc.removeQubit(2);
    EXPECT_EQ(p.first, nqubits);
    EXPECT_EQ(p.second, nqubits);
    EXPECT_EQ(qc.getNancillae(), 0);
    EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
    EXPECT_EQ(qc.getNqubits(), nqubits);
    EXPECT_TRUE(qc.getANCregs().empty());
    qc.printRegisters();
    qc.addAncillaryQubit(p.first, p.second);
    EXPECT_EQ(qc.getNancillae(), 1);
    EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
    EXPECT_EQ(qc.getNqubits(), nqubits + 1);
    EXPECT_FALSE(qc.getANCregs().empty());
    qc.printRegisters();
    auto q = qc.removeQubit(2);
    EXPECT_EQ(q.first, nqubits);
    EXPECT_EQ(q.second, nqubits);
    EXPECT_EQ(qc.getNancillae(), 0);
    EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
    EXPECT_EQ(qc.getNqubits(), nqubits);
    EXPECT_TRUE(qc.getANCregs().empty());
    qc.printRegisters();
    auto rm = qc.removeQubit(1);
    EXPECT_EQ(rm.first, 1);
    EXPECT_EQ(rm.second, 1);
    EXPECT_EQ(qc.getNancillae(), 0);
    EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 1);
    EXPECT_EQ(qc.getNqubits(), 1);
    qc.printRegisters();
    auto empty = qc.removeQubit(0);
    EXPECT_EQ(empty.first, 0);
    EXPECT_EQ(empty.second, 0);
    EXPECT_EQ(qc.getNancillae(), 0);
    EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 0);
    EXPECT_EQ(qc.getNqubits(), 0);
    EXPECT_TRUE(qc.getQregs().empty());
    qc.printRegisters();
    qc.printStatistics(std::cout);
}

TEST_F(QFRFunctionality, ancillary_qubit_remove_middle) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::X);
    qc.addAncillaryRegister(3);
    auto p = qc.removeQubit(3);
    EXPECT_EQ(p.first, 3);
    EXPECT_EQ(p.second, 3);
    EXPECT_EQ(qc.getNancillae(), 2);
    EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 2);
    EXPECT_EQ(qc.getNqubits(), 4);
    qc.printRegisters();
}

TEST_F(QFRFunctionality, split_qreg) {
    QubitCount         nqubits = 3;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::X);
    auto p = qc.removeQubit(1);
    EXPECT_EQ(p.first, 1);
    EXPECT_EQ(p.second, 1);
    EXPECT_EQ(qc.getNancillae(), 0);
    EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 2);
    EXPECT_EQ(qc.getNqubits(), 2);
    qc.printRegisters();
}

TEST_F(QFRFunctionality, FuseTwoSingleQubitGates) {
    QubitCount         nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::H);

    qc.print(std::cout);
    dd::Edge e = buildFunctionality(&qc, dd);
    CircuitOptimizer::singleQubitGateFusion(qc);
    dd::Edge f = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 1);
    EXPECT_EQ(e, f);
}

TEST_F(QFRFunctionality, FuseThreeSingleQubitGates) {
    QubitCount         nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::H);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::Y);

    dd::Edge e = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    dd::Edge f = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 1);
    EXPECT_EQ(e, f);
}

TEST_F(QFRFunctionality, FuseNoSingleQubitGates) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::H);
    qc.emplace_back<StandardOperation>(nqubits, 0_pc, 1, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::Y);
    dd::Edge e = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    dd::Edge f = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 3);
    EXPECT_EQ(e, f);
}

TEST_F(QFRFunctionality, FuseSingleQubitGatesAcrossOtherGates) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::H);
    qc.emplace_back<StandardOperation>(nqubits, 1, qc::Z);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::Y);
    auto e = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    auto f = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 2);
    EXPECT_EQ(e, f);
}

TEST_F(QFRFunctionality, StripIdleAndDump) {
    std::stringstream ss{};
    auto              testfile =
            "OPENQASM 2.0;\n"
            "include \"qelib1.inc\";\n"
            "qreg q[5];\n"
            "creg c[3];\n"
            "x q[0];\n"
            "x q[2];\n"
            "barrier q;\n"
            "barrier q[0];\n"
            "reset q;\n"
            "reset q[2];\n"
            "cx q[0],q[4];\n";

    ss << testfile;
    auto qc = qc::QuantumComputation();
    qc.import(ss, qc::OpenQASM);
    qc.print(std::cout);
    qc.stripIdleQubits();
    qc.print(std::cout);
    std::stringstream goal{};
    qc.print(goal);
    std::stringstream testss{};
    qc.dump(testss, OpenQASM);
    std::cout << testss.str() << std::endl;
    qc.reset();
    qc.import(testss, OpenQASM);
    qc.print(std::cout);
    qc.stripIdleQubits();
    qc.print(std::cout);
    std::stringstream actual{};
    qc.print(actual);
    EXPECT_EQ(goal.str(), actual.str());
}

TEST_F(QFRFunctionality, CollapseCompoundOperationToStandard) {
    QubitCount         nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::I);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 1);
    EXPECT_TRUE(qc.begin()->get()->isStandardOperation());
}

TEST_F(QFRFunctionality, eliminateCompoundOperation) {
    QubitCount         nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::I);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::I);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 0);
    EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, eliminateInverseInCompoundOperation) {
    QubitCount         nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, S);
    qc.emplace_back<StandardOperation>(nqubits, 0, Sdag);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 0);
    EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, unknownInverseInCompoundOperation) {
    QubitCount         nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, Phase, 1.);
    qc.emplace_back<StandardOperation>(nqubits, 0, Phase, -1.);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 1);
}

TEST_F(QFRFunctionality, removeDiagonalSingleQubitBeforeMeasure) {
    QubitCount         nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::Z);
    qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0}, std::vector<std::size_t>{0});
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 1);
    EXPECT_EQ(qc.begin()->get()->getType(), qc::Measure);
}

TEST_F(QFRFunctionality, removeDiagonalCompoundOpBeforeMeasure) {
    QubitCount         nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::Z);
    qc.emplace_back<StandardOperation>(nqubits, 0, T);
    qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0}, std::vector<std::size_t>{0});
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 1);
    EXPECT_EQ(qc.begin()->get()->getType(), qc::Measure);
}

TEST_F(QFRFunctionality, removeDiagonalTwoQubitGateBeforeMeasure) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0_pc, 1, qc::Z);
    qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0, 1}, std::vector<std::size_t>{0, 1});
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 1);
    EXPECT_EQ(qc.begin()->get()->getType(), qc::Measure);
}

TEST_F(QFRFunctionality, leaveGateBeforeMeasure) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0_pc, 1, qc::Z);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::X);
    qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0, 1}, std::vector<std::size_t>{0, 1});
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 3);
}

TEST_F(QFRFunctionality, removeComplexGateBeforeMeasure) {
    QubitCount         nqubits = 4;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0_pc, 1, qc::Z);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 1_pc, 2, qc::Z);
    qc.emplace_back<StandardOperation>(nqubits, 0_pc, 1, qc::Z);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::Z);
    qc.emplace_back<StandardOperation>(nqubits, 1_pc, 2, qc::Z);
    qc.emplace_back<StandardOperation>(nqubits, 3, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 3, qc::T);
    qc.emplace_back<StandardOperation>(nqubits, dd::Controls{0_pc, 1_pc, 2_pc}, 3, qc::Z);

    qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0, 1, 2, 3}, std::vector<std::size_t>{0, 1, 2, 3});
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 4);
}

TEST_F(QFRFunctionality, removeSimpleCompoundOpBeforeMeasure) {
    QubitCount         nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::T);
    qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0}, std::vector<std::size_t>{0});
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 2);
}

TEST_F(QFRFunctionality, removePartOfCompoundOpBeforeMeasure) {
    QubitCount         nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::T);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::X);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::T);
    qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0}, std::vector<std::size_t>{0});
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 2);
}

TEST_F(QFRFunctionality, decomposeSWAPsUndirectedArchitecture) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, Controls{}, 0, 1, SWAP);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::decomposeSWAP(qc, false);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    auto it = qc.begin();
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::X);
        EXPECT_EQ(op->getControls().begin()->qubit, 0);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::X);
        EXPECT_EQ(op->getControls().begin()->qubit, 1);
        EXPECT_EQ(op->getTargets().at(0), 0);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::X);
        EXPECT_EQ(op->getControls().begin()->qubit, 0);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
}
TEST_F(QFRFunctionality, decomposeSWAPsDirectedArchitecture) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, Controls{}, 0, 1, SWAP);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::decomposeSWAP(qc, true);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    auto it = qc.begin();
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::X);
        EXPECT_EQ(op->getControls().begin()->qubit, 0);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::H);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::H);
        EXPECT_EQ(op->getTargets().at(0), 0);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::X);
        EXPECT_EQ(op->getControls().begin()->qubit, 0);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::H);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::H);
        EXPECT_EQ(op->getTargets().at(0), 0);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::X);
        EXPECT_EQ(op->getControls().begin()->qubit, 0);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
}

TEST_F(QFRFunctionality, decomposeSWAPsCompound) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X); //We need something to replace with compound
    auto gate = std::make_unique<qc::CompoundOperation>(nqubits);
    gate->emplace_back<qc::StandardOperation>(nqubits, Controls{}, 0, 1, qc::SWAP);
    gate->emplace_back<qc::StandardOperation>(nqubits, Controls{}, 0, 1, qc::SWAP);
    gate->emplace_back<qc::StandardOperation>(nqubits, Controls{}, 0, 1, qc::SWAP);
    (*qc.begin()) = std::move(gate);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    qc::CircuitOptimizer::decomposeSWAP(qc, false);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    auto it = qc.begin();
    EXPECT_EQ(it->get()->isCompoundOperation(), true);

    EXPECT_EQ(dynamic_cast<qc::CompoundOperation*>(it->get())->size(), 9);
}
TEST_F(QFRFunctionality, decomposeSWAPsCompoundDirected) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X); //We need something to replace with compound
    auto gate = std::make_unique<qc::CompoundOperation>(nqubits);
    gate->emplace_back<qc::StandardOperation>(nqubits, Controls{}, 0, 1, qc::SWAP);
    gate->emplace_back<qc::StandardOperation>(nqubits, Controls{}, 0, 1, qc::SWAP);
    gate->emplace_back<qc::StandardOperation>(nqubits, Controls{}, 0, 1, qc::SWAP);
    (*qc.begin()) = std::move(gate);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    qc::CircuitOptimizer::decomposeSWAP(qc, true);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    auto it = qc.begin();
    EXPECT_EQ(it->get()->isCompoundOperation(), true);

    EXPECT_EQ(dynamic_cast<qc::CompoundOperation*>(it->get())->size(), 21);
}

TEST_F(QFRFunctionality, removeFinalMeasurements) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::H);
    qc.emplace_back<StandardOperation>(nqubits, 1, qc::H);
    qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0}, std::vector<std::size_t>{0});
    qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{1}, std::vector<std::size_t>{1});
    qc.emplace_back<StandardOperation>(nqubits, 1, qc::H);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::removeFinalMeasurements(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    auto it = qc.begin();
    ++it;
    ++it; //skip first two H
    ASSERT_NO_THROW({
        auto op = dynamic_cast<NonUnitaryOperation*>(it->get());
        EXPECT_EQ(op->getType(), Measure);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::H);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
}

TEST_F(QFRFunctionality, removeFinalMeasurementsTwoQubitMeasurement) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<StandardOperation>(nqubits, 0, qc::H);
    qc.emplace_back<StandardOperation>(nqubits, 1, qc::H);
    qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0, 1}, std::vector<std::size_t>{0, 1});
    qc.emplace_back<StandardOperation>(nqubits, 1, qc::H);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::removeFinalMeasurements(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    auto it = qc.begin();
    ++it;
    ++it; //skip first two H
    ASSERT_NO_THROW({
        auto op = dynamic_cast<NonUnitaryOperation*>(it->get());
        EXPECT_EQ(op->getType(), Measure);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::H);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
}

TEST_F(QFRFunctionality, removeFinalMeasurementsCompound) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X); //We need something to replace with compound
    auto gate = std::make_unique<qc::CompoundOperation>(nqubits);
    gate->emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0}, std::vector<std::size_t>{0});
    gate->emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{1}, std::vector<std::size_t>{1});
    gate->emplace_back<StandardOperation>(nqubits, 1, qc::H);
    (*qc.begin()) = std::move(gate);
    qc.emplace_back<StandardOperation>(nqubits, 1, qc::H);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::removeFinalMeasurements(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    auto it = qc.begin();
    EXPECT_EQ(it->get()->isCompoundOperation(), true);

    EXPECT_EQ(dynamic_cast<qc::CompoundOperation*>(it->get())->size(), 2);
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::H);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
}

TEST_F(QFRFunctionality, removeFinalMeasurementsCompoundDegraded) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X); //We need something to replace with compound
    auto gate = std::make_unique<qc::CompoundOperation>(nqubits);
    gate->emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0}, std::vector<std::size_t>{0});
    gate->emplace_back<StandardOperation>(nqubits, 1, qc::H);
    (*qc.begin()) = std::move(gate);
    qc.emplace_back<StandardOperation>(nqubits, 1, qc::H);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::removeFinalMeasurements(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    auto it = qc.begin();
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::H);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
    ++it;
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::H);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
}

TEST_F(QFRFunctionality, removeFinalMeasurementsCompoundEmpty) {
    QubitCount         nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X); //We need something to replace with compound
    auto gate = std::make_unique<qc::CompoundOperation>(nqubits);
    gate->emplace_back<NonUnitaryOperation>(nqubits, std::vector<dd::Qubit>{0}, std::vector<std::size_t>{0});
    (*qc.begin()) = std::move(gate);
    qc.emplace_back<StandardOperation>(nqubits, 1, qc::H);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::removeFinalMeasurements(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    auto it = qc.begin();
    ASSERT_NO_THROW({
        auto op = dynamic_cast<StandardOperation*>(it->get());
        EXPECT_EQ(op->getType(), qc::H);
        EXPECT_EQ(op->getTargets().at(0), 1);
    });
}

TEST_F(QFRFunctionality, removeFinalMeasurementsWithOperationsInFront) {
    auto              circ = "OPENQASM 2.0;include \"qelib1.inc\";qreg q[3];qreg r[3];h q;cx q, r;creg c[3];creg d[3];barrier q;measure q->c;measure r->d;\n";
    std::stringstream ss{};
    ss << circ;
    QuantumComputation qc{};
    qc.import(ss, qc::OpenQASM);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::removeFinalMeasurements(qc);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    ASSERT_EQ(qc.getNops(), 2);
    ASSERT_EQ(qc.getNindividualOps(), 6);
}

TEST_F(QFRFunctionality, gateShortCutsAndCloning) {
    QuantumComputation qc(5);
    qc.i(0);
    qc.i(0, 1_pc);
    qc.i(0, {1_pc, 2_nc});
    qc.h(0);
    qc.h(0, 1_pc);
    qc.h(0, {1_pc, 2_nc});
    qc.x(0);
    qc.x(0, 1_pc);
    qc.x(0, {1_pc, 2_nc});
    qc.y(0);
    qc.y(0, 1_pc);
    qc.y(0, {1_pc, 2_nc});
    qc.z(0);
    qc.z(0, 1_pc);
    qc.z(0, {1_pc, 2_nc});
    qc.s(0);
    qc.s(0, 1_pc);
    qc.s(0, {1_pc, 2_nc});
    qc.sdag(0);
    qc.sdag(0, 1_pc);
    qc.sdag(0, {1_pc, 2_nc});
    qc.t(0);
    qc.t(0, 1_pc);
    qc.t(0, {1_pc, 2_nc});
    qc.tdag(0);
    qc.tdag(0, 1_pc);
    qc.tdag(0, {1_pc, 2_nc});
    qc.v(0);
    qc.v(0, 1_pc);
    qc.v(0, {1_pc, 2_nc});
    qc.vdag(0);
    qc.vdag(0, 1_pc);
    qc.vdag(0, {1_pc, 2_nc});
    qc.u3(0, dd::PI, dd::PI, dd::PI);
    qc.u3(0, 1_pc, dd::PI, dd::PI, dd::PI);
    qc.u3(0, {1_pc, 2_nc}, dd::PI, dd::PI, dd::PI);
    qc.u2(0, dd::PI, dd::PI);
    qc.u2(0, 1_pc, dd::PI, dd::PI);
    qc.u2(0, {1_pc, 2_nc}, dd::PI, dd::PI);
    qc.phase(0, dd::PI);
    qc.phase(0, 1_pc, dd::PI);
    qc.phase(0, {1_pc, 2_nc}, dd::PI);
    qc.sx(0);
    qc.sx(0, 1_pc);
    qc.sx(0, {1_pc, 2_nc});
    qc.sxdag(0);
    qc.sxdag(0, 1_pc);
    qc.sxdag(0, {1_pc, 2_nc});
    qc.rx(0, dd::PI);
    qc.rx(0, 1_pc, dd::PI);
    qc.rx(0, {1_pc, 2_nc}, dd::PI);
    qc.ry(0, dd::PI);
    qc.ry(0, 1_pc, dd::PI);
    qc.ry(0, {1_pc, 2_nc}, dd::PI);
    qc.rz(0, dd::PI);
    qc.rz(0, 1_pc, dd::PI);
    qc.rz(0, {1_pc, 2_nc}, dd::PI);
    qc.swap(0, 1);
    qc.swap(0, 1, 2_pc);
    qc.swap(0, 1, {2_pc, 3_nc});
    qc.iswap(0, 1);
    qc.iswap(0, 1, 2_pc);
    qc.iswap(0, 1, {2_pc, 3_nc});
    qc.peres(0, 1);
    qc.peres(0, 1, 2_pc);
    qc.peres(0, 1, {2_pc, 3_nc});
    qc.peresdag(0, 1);
    qc.peresdag(0, 1, 2_pc);
    qc.peresdag(0, 1, {2_pc, 3_nc});
    qc.measure(0, 0);
    qc.measure({1, 2}, {1, 2});
    qc.barrier(0);
    qc.barrier({1, 2});
    qc.reset(0);
    qc.reset({1, 2});

    auto qc_cloned = qc.clone();
    ASSERT_EQ(qc.size(), qc_cloned.size());
}

TEST_F(QFRFunctionality, cloningDifferentOperations) {
    QuantumComputation qc(5);

    auto co = std::make_unique<CompoundOperation>(qc.getNqubits());
    co->emplace_back<NonUnitaryOperation>(co->getNqubits());
    co->emplace_back<StandardOperation>(co->getNqubits(), 0, qc::H);
    qc.emplace_back(co);
    std::unique_ptr<qc::Operation> op = std::make_unique<StandardOperation>(qc.getNqubits(), 0, qc::X);
    qc.emplace_back<ClassicControlledOperation>(op, qc.getCregs().at("c"), 1);
    qc.emplace_back<NonUnitaryOperation>(qc.getNqubits(), std::vector<dd::Qubit>{0, 1}, 1);

    auto qc_cloned = qc.clone();
    ASSERT_EQ(qc.size(), qc_cloned.size());
}

TEST_F(QFRFunctionality, wrongRegisterSizes) {
    QuantumComputation qc(5);
    ASSERT_THROW(qc.measure({0}, {1, 2}), std::invalid_argument);
}

TEST_F(QFRFunctionality, eliminateResetsBasicTest) {
    QuantumComputation qc{};
    qc.addQubitRegister(1);
    qc.addClassicalRegister(2);
    qc.h(0);
    qc.measure(0, 0U);
    qc.reset(0);
    qc.h(0);
    qc.measure(0, 1U);

    std::cout << qc << std::endl;

    EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

    EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

    std::cout << qc << std::endl;

    ASSERT_EQ(qc.getNqubits(), 2);
    ASSERT_EQ(qc.getNindividualOps(), 4);
    auto& op0 = qc.at(0);
    auto& op1 = qc.at(1);
    auto& op2 = qc.at(2);
    auto& op3 = qc.at(3);

    EXPECT_EQ(op0->getNqubits(), 2);
    EXPECT_TRUE(op0->getType() == qc::H);
    const auto& targets0 = op0->getTargets();
    EXPECT_EQ(targets0.size(), 1);
    EXPECT_EQ(targets0.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op0->getControls().empty());

    EXPECT_EQ(op1->getNqubits(), 2);
    EXPECT_TRUE(op1->getType() == qc::Measure);
    const auto& targets1 = op1->getTargets();
    EXPECT_EQ(targets1.size(), 1);
    EXPECT_EQ(targets1.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op1->getControls().empty());
    auto       measure0  = dynamic_cast<qc::NonUnitaryOperation*>(op1.get());
    const auto classics0 = measure0->getClassics();
    EXPECT_EQ(classics0.size(), 1);
    EXPECT_EQ(classics0.at(0), 0);

    EXPECT_EQ(op2->getNqubits(), 2);
    EXPECT_TRUE(op2->getType() == qc::H);
    const auto& targets2 = op2->getTargets();
    EXPECT_EQ(targets2.size(), 1);
    EXPECT_EQ(targets2.at(0), static_cast<dd::Qubit>(1));
    EXPECT_TRUE(op2->getControls().empty());

    EXPECT_EQ(op3->getNqubits(), 2);
    EXPECT_TRUE(op3->getType() == qc::Measure);
    const auto& targets3 = op3->getTargets();
    EXPECT_EQ(targets3.size(), 1);
    EXPECT_EQ(targets3.at(0), static_cast<dd::Qubit>(1));
    EXPECT_TRUE(op3->getControls().empty());
    auto       measure1  = dynamic_cast<qc::NonUnitaryOperation*>(op3.get());
    const auto classics1 = measure1->getClassics();
    EXPECT_EQ(classics1.size(), 1);
    EXPECT_EQ(classics1.at(0), 1);
}

TEST_F(QFRFunctionality, eliminateResetsClassicControlled) {
    QuantumComputation qc{};
    qc.addQubitRegister(1);
    qc.addClassicalRegister(2);
    qc.h(0);
    qc.measure(0, 0U);
    qc.reset(0);
    std::unique_ptr<qc::Operation> xOp = std::make_unique<qc::StandardOperation>(1U, 0, qc::X);
    qc.emplace_back<qc::ClassicControlledOperation>(xOp, std::pair{0, 1U}, 1U);

    std::cout << qc << std::endl;

    EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

    EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

    std::cout << qc << std::endl;

    ASSERT_EQ(qc.getNqubits(), 2);
    ASSERT_EQ(qc.getNindividualOps(), 3);
    auto& op0 = qc.at(0);
    auto& op1 = qc.at(1);
    auto& op2 = qc.at(2);

    EXPECT_EQ(op0->getNqubits(), 2);
    EXPECT_TRUE(op0->getType() == qc::H);
    const auto& targets0 = op0->getTargets();
    EXPECT_EQ(targets0.size(), 1);
    EXPECT_EQ(targets0.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op0->getControls().empty());

    EXPECT_EQ(op1->getNqubits(), 2);
    EXPECT_TRUE(op1->getType() == qc::Measure);
    const auto& targets1 = op1->getTargets();
    EXPECT_EQ(targets1.size(), 1);
    EXPECT_EQ(targets1.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op1->getControls().empty());
    auto       measure0  = dynamic_cast<qc::NonUnitaryOperation*>(op1.get());
    const auto classics0 = measure0->getClassics();
    EXPECT_EQ(classics0.size(), 1);
    EXPECT_EQ(classics0.at(0), 0);

    EXPECT_EQ(op2->getNqubits(), 2);
    EXPECT_TRUE(op2->isClassicControlledOperation());
    auto       classicControlled = dynamic_cast<qc::ClassicControlledOperation*>(op2.get());
    const auto operation         = classicControlled->getOperation();
    EXPECT_EQ(operation->getNqubits(), 2);
    EXPECT_TRUE(operation->getType() == qc::X);
    EXPECT_EQ(classicControlled->getNtargets(), 1);
    const auto& targets = classicControlled->getTargets();
    EXPECT_EQ(targets.at(0), 1);
    EXPECT_EQ(classicControlled->getNcontrols(), 0);
}

TEST_F(QFRFunctionality, eliminateResetsMultipleTargetReset) {
    QuantumComputation qc{};
    qc.addQubitRegister(2);
    qc.reset({0, 1});
    qc.x(0);
    qc.z(1);
    qc.x(0, 1_pc);

    std::cout << qc << std::endl;

    EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

    EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

    std::cout << qc << std::endl;

    ASSERT_EQ(qc.getNqubits(), 4);
    ASSERT_EQ(qc.getNindividualOps(), 3);
    auto& op0 = qc.at(0);
    auto& op1 = qc.at(1);
    auto& op2 = qc.at(2);

    EXPECT_EQ(op0->getNqubits(), 4);
    EXPECT_TRUE(op0->getType() == qc::X);
    const auto& targets0 = op0->getTargets();
    EXPECT_EQ(targets0.size(), 1);
    EXPECT_EQ(targets0.at(0), static_cast<dd::Qubit>(2));
    EXPECT_TRUE(op0->getControls().empty());

    EXPECT_EQ(op1->getNqubits(), 4);
    EXPECT_TRUE(op1->getType() == qc::Z);
    const auto& targets1 = op1->getTargets();
    EXPECT_EQ(targets1.size(), 1);
    EXPECT_EQ(targets1.at(0), static_cast<dd::Qubit>(3));
    EXPECT_TRUE(op1->getControls().empty());

    EXPECT_EQ(op2->getNqubits(), 4);
    EXPECT_TRUE(op2->getType() == qc::X);
    const auto& targets2 = op2->getTargets();
    EXPECT_EQ(targets2.size(), 1);
    EXPECT_EQ(targets2.at(0), static_cast<dd::Qubit>(2));
    const auto& controls2 = op2->getControls();
    EXPECT_EQ(controls2.size(), 1);
    EXPECT_EQ(controls2.count(3), 1);
}

TEST_F(QFRFunctionality, eliminateResetsCompoundOperation) {
    QuantumComputation qc(2U);

    qc.reset(0);
    qc.reset(1);

    qc.emplace_back<qc::CompoundOperation>(qc.getNqubits());
    auto compOp = dynamic_cast<qc::CompoundOperation*>(qc.at(2).get());
    compOp->emplace_back<qc::StandardOperation>(2U, 1_pc, 0, qc::X);
    compOp->emplace_back<qc::NonUnitaryOperation>(2U, std::vector<dd::Qubit>{0}, qc::Reset);
    compOp->emplace_back<qc::NonUnitaryOperation>(2U, 0, 0U);
    std::unique_ptr<qc::Operation> xOp = std::make_unique<qc::StandardOperation>(2U, 0, qc::X);
    compOp->emplace_back<qc::ClassicControlledOperation>(xOp, std::pair{0, 1U}, 1U);

    std::cout << qc << std::endl;

    EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

    EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

    std::cout << qc << std::endl;

    ASSERT_EQ(qc.getNqubits(), 5);
    ASSERT_EQ(qc.getNindividualOps(), 3);

    auto& op = qc.at(0);
    EXPECT_TRUE(op->isCompoundOperation());
    EXPECT_EQ(op->getNqubits(), 5);
    auto compOp0 = dynamic_cast<qc::CompoundOperation*>(op.get());
    EXPECT_EQ(compOp0->size(), 3);

    auto& op0 = compOp0->at(0);
    auto& op1 = compOp0->at(1);
    auto& op2 = compOp0->at(2);

    EXPECT_EQ(op0->getNqubits(), 5);
    EXPECT_TRUE(op0->getType() == qc::X);
    const auto& targets0 = op0->getTargets();
    EXPECT_EQ(targets0.size(), 1);
    EXPECT_EQ(targets0.at(0), static_cast<dd::Qubit>(2));
    const auto& controls0 = op0->getControls();
    EXPECT_EQ(controls0.size(), 1);
    EXPECT_EQ(controls0.count(3), 1);

    EXPECT_EQ(op1->getNqubits(), 5);
    EXPECT_TRUE(op1->getType() == qc::Measure);
    const auto& targets1 = op1->getTargets();
    EXPECT_EQ(targets1.size(), 1);
    EXPECT_EQ(targets1.at(0), static_cast<dd::Qubit>(4));
    EXPECT_TRUE(op1->getControls().empty());
    auto       measure0  = dynamic_cast<qc::NonUnitaryOperation*>(op1.get());
    const auto classics0 = measure0->getClassics();
    EXPECT_EQ(classics0.size(), 1);
    EXPECT_EQ(classics0.at(0), 0);

    EXPECT_EQ(op2->getNqubits(), 5);
    EXPECT_TRUE(op2->isClassicControlledOperation());
    auto       classicControlled = dynamic_cast<qc::ClassicControlledOperation*>(op2.get());
    const auto operation         = classicControlled->getOperation();
    EXPECT_EQ(operation->getNqubits(), 5);
    EXPECT_TRUE(operation->getType() == qc::X);
    EXPECT_EQ(classicControlled->getNtargets(), 1);
    const auto& targets = classicControlled->getTargets();
    EXPECT_EQ(targets.at(0), 4);
    EXPECT_EQ(classicControlled->getNcontrols(), 0);
}

TEST_F(QFRFunctionality, deferMeasurementsBasicTest) {
    // Input:
    // i: 			0	1
    //1: 	H   	H 	|
    //2: 	Meas	0	|
    //3: 	c_X   	|	X 		c[0] == 1
    //o: 			0	1

    // Expected Output:
    // i: 			0	1
    //1: 	H   	H 	|
    //3: 	X   	c	X
    //2: 	Meas	0	|
    //o: 			0	1

    QuantumComputation qc{};
    qc.addQubitRegister(2);
    qc.addClassicalRegister(1);
    qc.h(0);
    qc.measure(0, 0U);
    std::unique_ptr<qc::Operation> xOp = std::make_unique<qc::StandardOperation>(2U, 1, qc::X);
    qc.emplace_back<qc::ClassicControlledOperation>(xOp, std::pair{0, 1U}, 1U);

    std::cout << qc << std::endl;

    EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

    EXPECT_NO_THROW(CircuitOptimizer::deferMeasurements(qc););

    std::cout << qc << std::endl;

    EXPECT_FALSE(CircuitOptimizer::isDynamicCircuit(qc));

    ASSERT_EQ(qc.getNqubits(), 2);
    ASSERT_EQ(qc.getNindividualOps(), 3);
    auto& op0 = qc.at(0);
    auto& op1 = qc.at(1);
    auto& op2 = qc.at(2);

    EXPECT_EQ(op0->getNqubits(), 2);
    EXPECT_TRUE(op0->getType() == qc::H);
    const auto& targets0 = op0->getTargets();
    EXPECT_EQ(targets0.size(), 1);
    EXPECT_EQ(targets0.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op0->getControls().empty());

    EXPECT_EQ(op1->getNqubits(), 2);
    EXPECT_TRUE(op1->getType() == qc::X);
    const auto& targets1 = op1->getTargets();
    EXPECT_EQ(targets1.size(), 1);
    EXPECT_EQ(targets1.at(0), static_cast<dd::Qubit>(1));
    const auto& controls1 = op1->getControls();
    EXPECT_EQ(controls1.size(), 1);
    EXPECT_EQ(controls1.count(0), 1);

    EXPECT_EQ(op2->getNqubits(), 2);
    ASSERT_TRUE(op2->getType() == qc::Measure);
    const auto& targets2 = op2->getTargets();
    EXPECT_EQ(targets2.size(), 1);
    EXPECT_EQ(targets2.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op2->getControls().empty());
    auto       measure0  = dynamic_cast<qc::NonUnitaryOperation*>(op2.get());
    const auto classics0 = measure0->getClassics();
    EXPECT_EQ(classics0.size(), 1);
    EXPECT_EQ(classics0.at(0), 0);
}

TEST_F(QFRFunctionality, deferMeasurementsOperationBetweenMeasurementAndClassic) {
    // Input:
    // i: 			0	1
    //1: 	H   	H 	|
    //2: 	Meas	0	|
    //3: 	H   	H 	|
    //4: 	c_X   	|	X 		c[0] == 1
    //o: 			0	1

    // Expected Output:
    // i: 			0	1
    //1: 	H   	H 	|
    //2: 	X   	c	X
    //3: 	H   	H 	|
    //4: 	Meas	0	|
    //o: 			0	1

    QuantumComputation qc{};
    qc.addQubitRegister(2);
    qc.addClassicalRegister(1);
    qc.h(0);
    qc.measure(0, 0U);
    qc.h(0);
    std::unique_ptr<qc::Operation> xOp = std::make_unique<qc::StandardOperation>(2U, 1, qc::X);
    qc.emplace_back<qc::ClassicControlledOperation>(xOp, std::pair{0, 1U}, 1U);

    std::cout << qc << std::endl;

    EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

    EXPECT_NO_THROW(CircuitOptimizer::deferMeasurements(qc););

    std::cout << qc << std::endl;

    EXPECT_FALSE(CircuitOptimizer::isDynamicCircuit(qc));

    ASSERT_EQ(qc.getNqubits(), 2);
    ASSERT_EQ(qc.getNindividualOps(), 4);
    auto& op0 = qc.at(0);
    auto& op1 = qc.at(1);
    auto& op2 = qc.at(2);
    auto& op3 = qc.at(3);

    EXPECT_EQ(op0->getNqubits(), 2);
    EXPECT_TRUE(op0->getType() == qc::H);
    const auto& targets0 = op0->getTargets();
    EXPECT_EQ(targets0.size(), 1);
    EXPECT_EQ(targets0.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op0->getControls().empty());

    EXPECT_EQ(op1->getNqubits(), 2);
    EXPECT_TRUE(op1->getType() == qc::X);
    const auto& targets1 = op1->getTargets();
    EXPECT_EQ(targets1.size(), 1);
    EXPECT_EQ(targets1.at(0), static_cast<dd::Qubit>(1));
    const auto& controls1 = op1->getControls();
    EXPECT_EQ(controls1.size(), 1);
    EXPECT_EQ(controls1.count(0), 1);

    EXPECT_EQ(op2->getNqubits(), 2);
    EXPECT_TRUE(op2->getType() == qc::H);
    const auto& targets2 = op2->getTargets();
    EXPECT_EQ(targets2.size(), 1);
    EXPECT_EQ(targets2.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op2->getControls().empty());

    EXPECT_EQ(op3->getNqubits(), 2);
    ASSERT_TRUE(op3->getType() == qc::Measure);
    const auto& targets3 = op3->getTargets();
    EXPECT_EQ(targets3.size(), 1);
    EXPECT_EQ(targets3.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op3->getControls().empty());
    auto       measure0  = dynamic_cast<qc::NonUnitaryOperation*>(op3.get());
    const auto classics0 = measure0->getClassics();
    EXPECT_EQ(classics0.size(), 1);
    EXPECT_EQ(classics0.at(0), 0);
}

TEST_F(QFRFunctionality, deferMeasurementsTwoClassic) {
    // Input:
    // i: 			0	1
    //1: 	H   	H 	|
    //2: 	Meas	0	|
    //3: 	H   	H 	|
    //4: 	c_X   	|	X 		c[0] == 1
    //5:    c_Z     |   Z       c[0] == 1
    //o: 			0	1

    // Expected Output:
    // i: 			0	1
    //1: 	H   	H 	|
    //2: 	X   	c	X
    //3: 	Z   	c	Z
    //4: 	H   	H 	|
    //5: 	Meas	0	|
    //o: 			0	1

    QuantumComputation qc{};
    qc.addQubitRegister(2);
    qc.addClassicalRegister(1);
    qc.h(0);
    qc.measure(0, 0U);
    qc.h(0);
    std::unique_ptr<qc::Operation> xOp = std::make_unique<qc::StandardOperation>(2U, 1, qc::X);
    qc.emplace_back<qc::ClassicControlledOperation>(xOp, std::pair{0, 1U}, 1U);
    std::unique_ptr<qc::Operation> zOp = std::make_unique<qc::StandardOperation>(2U, 1, qc::Z);
    qc.emplace_back<qc::ClassicControlledOperation>(zOp, std::pair{0, 1U}, 1U);

    std::cout << qc << std::endl;

    EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

    EXPECT_NO_THROW(CircuitOptimizer::deferMeasurements(qc););

    std::cout << qc << std::endl;

    EXPECT_FALSE(CircuitOptimizer::isDynamicCircuit(qc));

    ASSERT_EQ(qc.getNqubits(), 2);
    ASSERT_EQ(qc.getNindividualOps(), 5);
    auto& op0 = qc.at(0);
    auto& op1 = qc.at(1);
    auto& op2 = qc.at(2);
    auto& op3 = qc.at(3);
    auto& op4 = qc.at(4);

    EXPECT_EQ(op0->getNqubits(), 2);
    EXPECT_TRUE(op0->getType() == qc::H);
    const auto& targets0 = op0->getTargets();
    EXPECT_EQ(targets0.size(), 1);
    EXPECT_EQ(targets0.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op0->getControls().empty());

    EXPECT_EQ(op1->getNqubits(), 2);
    EXPECT_TRUE(op1->getType() == qc::X);
    const auto& targets1 = op1->getTargets();
    EXPECT_EQ(targets1.size(), 1);
    EXPECT_EQ(targets1.at(0), static_cast<dd::Qubit>(1));
    const auto& controls1 = op1->getControls();
    EXPECT_EQ(controls1.size(), 1);
    EXPECT_EQ(controls1.count(0), 1);

    EXPECT_EQ(op2->getNqubits(), 2);
    EXPECT_TRUE(op2->getType() == qc::Z);
    const auto& targets2 = op2->getTargets();
    EXPECT_EQ(targets2.size(), 1);
    EXPECT_EQ(targets2.at(0), static_cast<dd::Qubit>(1));
    const auto& controls2 = op2->getControls();
    EXPECT_EQ(controls2.size(), 1);
    EXPECT_EQ(controls2.count(0), 1);

    EXPECT_EQ(op3->getNqubits(), 2);
    EXPECT_TRUE(op3->getType() == qc::H);
    const auto& targets3 = op3->getTargets();
    EXPECT_EQ(targets3.size(), 1);
    EXPECT_EQ(targets3.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op3->getControls().empty());

    EXPECT_EQ(op4->getNqubits(), 2);
    ASSERT_TRUE(op4->getType() == qc::Measure);
    const auto& targets4 = op4->getTargets();
    EXPECT_EQ(targets4.size(), 1);
    EXPECT_EQ(targets4.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op4->getControls().empty());
    auto       measure0  = dynamic_cast<qc::NonUnitaryOperation*>(op4.get());
    const auto classics0 = measure0->getClassics();
    EXPECT_EQ(classics0.size(), 1);
    EXPECT_EQ(classics0.at(0), 0);
}

TEST_F(QFRFunctionality, deferMeasurementsCorrectOrder) {
    // Input:
    // i: 			0	1
    //1: 	H   	H 	|
    //2: 	Meas	0	|
    //3: 	H   	| 	H
    //4: 	c_X   	|	X 		c[0] == 1
    //o: 			0	1

    // Expected Output:
    // i: 			0	1
    //1: 	H   	H 	|
    //2: 	H   	| 	H
    //3: 	X   	c	X
    //4: 	Meas	0	|
    //o: 			0	1

    QuantumComputation qc{};
    qc.addQubitRegister(2);
    qc.addClassicalRegister(1);
    qc.h(0);
    qc.measure(0, 0U);
    qc.h(1);
    std::unique_ptr<qc::Operation> xOp = std::make_unique<qc::StandardOperation>(2U, 1, qc::X);
    qc.emplace_back<qc::ClassicControlledOperation>(xOp, std::pair{0, 1U}, 1U);

    std::cout << qc << std::endl;

    EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

    EXPECT_NO_THROW(CircuitOptimizer::deferMeasurements(qc););

    std::cout << qc << std::endl;

    EXPECT_FALSE(CircuitOptimizer::isDynamicCircuit(qc));

    ASSERT_EQ(qc.getNqubits(), 2);
    ASSERT_EQ(qc.getNindividualOps(), 4);
    auto& op0 = qc.at(0);
    auto& op1 = qc.at(1);
    auto& op2 = qc.at(2);
    auto& op3 = qc.at(3);

    EXPECT_EQ(op0->getNqubits(), 2);
    EXPECT_TRUE(op0->getType() == qc::H);
    const auto& targets0 = op0->getTargets();
    EXPECT_EQ(targets0.size(), 1);
    EXPECT_EQ(targets0.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op0->getControls().empty());

    EXPECT_EQ(op1->getNqubits(), 2);
    EXPECT_TRUE(op1->getType() == qc::H);
    const auto& targets1 = op2->getTargets();
    EXPECT_EQ(targets1.size(), 1);
    EXPECT_EQ(targets1.at(0), static_cast<dd::Qubit>(1));
    EXPECT_TRUE(op1->getControls().empty());

    EXPECT_EQ(op2->getNqubits(), 2);
    EXPECT_TRUE(op2->getType() == qc::X);
    const auto& targets2 = op1->getTargets();
    EXPECT_EQ(targets2.size(), 1);
    EXPECT_EQ(targets2.at(0), static_cast<dd::Qubit>(1));
    const auto& controls2 = op2->getControls();
    EXPECT_EQ(controls2.size(), 1);
    EXPECT_EQ(controls2.count(0), 1);

    EXPECT_EQ(op3->getNqubits(), 2);
    ASSERT_TRUE(op3->getType() == qc::Measure);
    const auto& targets3 = op3->getTargets();
    EXPECT_EQ(targets3.size(), 1);
    EXPECT_EQ(targets3.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op3->getControls().empty());
    auto       measure0  = dynamic_cast<qc::NonUnitaryOperation*>(op3.get());
    const auto classics0 = measure0->getClassics();
    EXPECT_EQ(classics0.size(), 1);
    EXPECT_EQ(classics0.at(0), 0);
}

TEST_F(QFRFunctionality, deferMeasurementsTwoClassicCorrectOrder) {
    // Input:
    // i: 			0	1
    //1: 	H   	H 	|
    //2: 	Meas	0	|
    //3: 	H   	| 	H
    //4: 	c_X   	|	X 		c[0] == 1
    //5:    c_Z     |   Z       c[0] == 1
    //o: 			0	1

    // Expected Output:
    // i: 			0	1
    //1: 	H   	H 	|
    //2: 	H   	| 	H
    //3: 	X   	c	X
    //4: 	Z   	c	Z
    //5: 	Meas	0	|
    //o: 			0	1

    QuantumComputation qc{};
    qc.addQubitRegister(2);
    qc.addClassicalRegister(1);
    qc.h(0);
    qc.measure(0, 0U);
    qc.h(1);
    std::unique_ptr<qc::Operation> xOp = std::make_unique<qc::StandardOperation>(2U, 1, qc::X);
    qc.emplace_back<qc::ClassicControlledOperation>(xOp, std::pair{0, 1U}, 1U);
    std::unique_ptr<qc::Operation> zOp = std::make_unique<qc::StandardOperation>(2U, 1, qc::Z);
    qc.emplace_back<qc::ClassicControlledOperation>(zOp, std::pair{0, 1U}, 1U);

    std::cout << qc << std::endl;

    EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

    EXPECT_NO_THROW(CircuitOptimizer::deferMeasurements(qc););

    std::cout << qc << std::endl;

    EXPECT_FALSE(CircuitOptimizer::isDynamicCircuit(qc));

    ASSERT_EQ(qc.getNqubits(), 2);
    ASSERT_EQ(qc.getNindividualOps(), 5);
    auto& op0 = qc.at(0);
    auto& op1 = qc.at(1);
    auto& op2 = qc.at(2);
    auto& op3 = qc.at(3);
    auto& op4 = qc.at(4);

    EXPECT_EQ(op0->getNqubits(), 2);
    EXPECT_TRUE(op0->getType() == qc::H);
    const auto& targets0 = op0->getTargets();
    EXPECT_EQ(targets0.size(), 1);
    EXPECT_EQ(targets0.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op0->getControls().empty());

    EXPECT_EQ(op1->getNqubits(), 2);
    EXPECT_TRUE(op1->getType() == qc::H);
    const auto& targets1 = op1->getTargets();
    EXPECT_EQ(targets1.size(), 1);
    EXPECT_EQ(targets1.at(0), static_cast<dd::Qubit>(1));
    EXPECT_TRUE(op1->getControls().empty());

    EXPECT_EQ(op2->getNqubits(), 2);
    EXPECT_TRUE(op2->getType() == qc::X);
    const auto& targets2 = op2->getTargets();
    EXPECT_EQ(targets2.size(), 1);
    EXPECT_EQ(targets2.at(0), static_cast<dd::Qubit>(1));
    const auto& controls2 = op2->getControls();
    EXPECT_EQ(controls2.size(), 1);
    EXPECT_EQ(controls2.count(0), 1);

    EXPECT_EQ(op3->getNqubits(), 2);
    EXPECT_TRUE(op3->getType() == qc::Z);
    const auto& targets3 = op3->getTargets();
    EXPECT_EQ(targets3.size(), 1);
    EXPECT_EQ(targets3.at(0), static_cast<dd::Qubit>(1));
    const auto& controls3 = op3->getControls();
    EXPECT_EQ(controls3.size(), 1);
    EXPECT_EQ(controls3.count(0), 1);

    EXPECT_EQ(op4->getNqubits(), 2);
    ASSERT_TRUE(op4->getType() == qc::Measure);
    const auto& targets4 = op4->getTargets();
    EXPECT_EQ(targets4.size(), 1);
    EXPECT_EQ(targets4.at(0), static_cast<dd::Qubit>(0));
    EXPECT_TRUE(op4->getControls().empty());
    auto       measure0  = dynamic_cast<qc::NonUnitaryOperation*>(op4.get());
    const auto classics0 = measure0->getClassics();
    EXPECT_EQ(classics0.size(), 1);
    EXPECT_EQ(classics0.at(0), 0);
}

TEST_F(QFRFunctionality, deferMeasurementsErrorOnImplicitReset) {
    // Input:
    // i: 			0
    //1: 	H   	H
    //2: 	Meas	0
    //3: 	c_X   	X	c[0] == 1
    //o: 			0

    // Expected Output:
    // Error, since the classic-controlled operation targets the qubit being measured (this implicitly realizes a reset operation)

    QuantumComputation qc{1};
    qc.h(0);
    qc.measure(0, 0U);
    std::unique_ptr<qc::Operation> xOp = std::make_unique<qc::StandardOperation>(1U, 0, qc::X);
    qc.emplace_back<qc::ClassicControlledOperation>(xOp, std::pair{0, 1U}, 1U);

    std::cout << qc << std::endl;

    EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

    EXPECT_THROW(CircuitOptimizer::deferMeasurements(qc), qc::QFRException);
}

TEST_F(QFRFunctionality, basicTensorDumpTest) {
    QuantumComputation qc(2);
    qc.h(1);
    qc.x(0, 1_pc);

    std::stringstream ss{};
    qc.dump(ss, qc::Tensor);

    auto reference = "{\"tensors\": [\n"
                     "[[\"H   \", \"Q1\", \"GATE0\"], [\"q1_0\", \"q1_1\"], [2, 2], [[0.70710678118654757, 0], [0.70710678118654757, 0], [0.70710678118654757, 0], [-0.70710678118654757, 0]]],\n"
                     "[[\"X   \", \"Q1\", \"Q0\", \"GATE1\"], [\"q1_1\", \"q0_0\", \"q1_2\", \"q0_1\"], [2, 2, 2, 2], [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [1, 0], [0, 0]]]\n"
                     "]}\n";
    EXPECT_EQ(ss.str(), reference);
}

TEST_F(QFRFunctionality, compoundTensorDumpTest) {
    QuantumComputation qc(2);
    qc.emplace_back<qc::CompoundOperation>(3);
    auto compop = dynamic_cast<qc::CompoundOperation*>(qc.begin()->get());
    compop->emplace_back<qc::StandardOperation>(2, 1, qc::H);
    compop->emplace_back<qc::StandardOperation>(2, 1_pc, 0, qc::X);

    std::stringstream ss{};
    qc.dump(ss, qc::Tensor);

    auto reference = "{\"tensors\": [\n"
                     "[[\"H   \", \"Q1\", \"GATE0\"], [\"q1_0\", \"q1_1\"], [2, 2], [[0.70710678118654757, 0], [0.70710678118654757, 0], [0.70710678118654757, 0], [-0.70710678118654757, 0]]],\n"
                     "[[\"X   \", \"Q1\", \"Q0\", \"GATE1\"], [\"q1_1\", \"q0_0\", \"q1_2\", \"q0_1\"], [2, 2, 2, 2], [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [1, 0], [0, 0]]]\n"
                     "]}\n";
    EXPECT_EQ(ss.str(), reference);
}

TEST_F(QFRFunctionality, errorTensorDumpTest) {
    QuantumComputation             qc(2);
    std::unique_ptr<qc::Operation> op = std::make_unique<qc::StandardOperation>(2, 0, qc::X);
    qc.emplace_back<qc::ClassicControlledOperation>(op, std::pair{0, 1U}, 1U);

    std::stringstream ss{};
    EXPECT_THROW(qc.dump(ss, qc::Tensor), qc::QFRException);

    ss.str("");
    qc.erase(qc.begin());
    qc.barrier(0);
    qc.measure(0, 0);
    EXPECT_NO_THROW(qc.dump(ss, qc::Tensor));

    ss.str("");
    qc.reset(0);
    EXPECT_THROW(qc.dump(ss, qc::Tensor), qc::QFRException);
}

TEST_F(QFRFunctionality, trivialOperationReordering) {
    QuantumComputation qc(2);
    qc.h(0);
    qc.h(1);
    std::cout << qc << std::endl;
    qc::CircuitOptimizer::reorderOperations(qc);
    std::cout << qc << std::endl;
    auto       it     = qc.begin();
    const auto target = (*it)->getTargets().at(0);
    EXPECT_EQ(target, 1);
    ++it;
    const auto target2 = (*it)->getTargets().at(0);
    EXPECT_EQ(target2, 0);
}

TEST_F(QFRFunctionality, FlattenRandomClifford) {
    qc::RandomCliffordCircuit rcs(2U, 3U, 0U);
    std::cout << rcs << std::endl;

    auto dd     = std::make_unique<dd::Package<>>(2U);
    auto before = buildFunctionality(&rcs, dd);

    qc::CircuitOptimizer::flattenOperations(rcs);
    std::cout << rcs << std::endl;

    for (const auto& op: rcs) {
        EXPECT_FALSE(op->isCompoundOperation());
    }

    auto after = buildFunctionality(&rcs, dd);
    EXPECT_EQ(before, after);
}

TEST_F(QFRFunctionality, FlattenRecursive) {
    const dd::QubitCount nqubits = 1U;

    // create a compound operation
    auto op = std::make_unique<CompoundOperation>(nqubits);

    // emplace an operation in the compound operation
    op->emplace_back<StandardOperation>(1U, 0U, qc::X);

    // create another compound operation
    auto op2 = std::make_unique<CompoundOperation>(nqubits);

    // emplace the first operation in the second
    op2->emplace_back<qc::CompoundOperation>(op);

    // create a quantum computation and emplace the operation
    auto qc = QuantumComputation(nqubits);
    qc.emplace_back<CompoundOperation>(op2);

    std::cout << qc << std::endl;

    qc::CircuitOptimizer::flattenOperations(qc);
    std::cout << qc << std::endl;

    for (const auto& g: qc) {
        EXPECT_FALSE(g->isCompoundOperation());
    }

    auto& gate = **qc.begin();
    EXPECT_EQ(gate.getType(), qc::X);
    EXPECT_EQ(gate.getTargets().at(0), 0U);
    EXPECT_TRUE(gate.getControls().empty());
}

TEST_F(QFRFunctionality, OperationEquality) {
    const auto x = StandardOperation(1U, 0, qc::X);
    const auto z = StandardOperation(1U, 0, qc::Z);
    EXPECT_TRUE(x.equals(x));
    EXPECT_FALSE(x.equals(z));

    const auto x0 = StandardOperation(2U, 0, qc::X);
    const auto x1 = StandardOperation(2U, 1, qc::X);
    EXPECT_FALSE(x0.equals(x1));
    Permutation perm0{};
    perm0[0] = 1;
    perm0[1] = 0;
    EXPECT_TRUE(x0.equals(x1, perm0, {}));
    EXPECT_TRUE(x0.equals(x1, {}, perm0));

    const auto cx01 = StandardOperation(2U, 0_pc, 1, qc::X);
    const auto cx10 = StandardOperation(2U, 1_pc, 0, qc::X);
    EXPECT_FALSE(cx01.equals(cx10));
    EXPECT_FALSE(x0.equals(cx01));

    const auto p  = StandardOperation(1U, 0, qc::Phase, 2.0);
    const auto pm = StandardOperation(1U, 0, qc::Phase, -2.0);
    EXPECT_FALSE(p.equals(pm));

    const auto measure0 = NonUnitaryOperation(2U, 0, 0U);
    const auto measure1 = NonUnitaryOperation(2U, 0, 1U);
    const auto measure2 = NonUnitaryOperation(2U, 1, 0U);
    EXPECT_FALSE(measure0.equals(x0));
    EXPECT_TRUE(measure0.equals(measure0));
    EXPECT_FALSE(measure0.equals(measure1));
    EXPECT_FALSE(measure0.equals(measure2));
    EXPECT_TRUE(measure0.equals(measure2, perm0, {}));
    EXPECT_TRUE(measure0.equals(measure2, {}, perm0));

    const auto controlRegister0 = qc::QuantumRegister{0, 1U};
    const auto controlRegister1 = qc::QuantumRegister{1, 1U};
    const auto expectedValue0   = 0U;
    const auto expectedValue1   = 1U;

    std::unique_ptr<Operation> xp0      = std::make_unique<StandardOperation>(1U, 0, qc::X);
    std::unique_ptr<Operation> xp1      = std::make_unique<StandardOperation>(1U, 0, qc::X);
    std::unique_ptr<Operation> xp2      = std::make_unique<StandardOperation>(1U, 0, qc::X);
    const auto                 classic0 = ClassicControlledOperation(xp0, controlRegister0, expectedValue0);
    const auto                 classic1 = ClassicControlledOperation(xp1, controlRegister0, expectedValue1);
    const auto                 classic2 = ClassicControlledOperation(xp2, controlRegister1, expectedValue0);
    std::unique_ptr<Operation> zp       = std::make_unique<StandardOperation>(1U, 0, qc::Z);
    const auto                 classic3 = ClassicControlledOperation(zp, controlRegister0, expectedValue0);
    EXPECT_FALSE(classic0.equals(x));
    EXPECT_TRUE(classic0.equals(classic0));
    EXPECT_FALSE(classic0.equals(classic1));
    EXPECT_FALSE(classic0.equals(classic2));
    EXPECT_FALSE(classic0.equals(classic3));

    auto compound0 = CompoundOperation(1U);
    compound0.emplace_back<StandardOperation>(1U, 0, qc::X);

    auto compound1 = CompoundOperation(1U);
    compound1.emplace_back<StandardOperation>(1U, 0, qc::X);
    compound1.emplace_back<StandardOperation>(1U, 0, qc::Z);

    auto compound2 = CompoundOperation(1U);
    compound2.emplace_back<StandardOperation>(1U, 0, qc::Z);

    EXPECT_FALSE(compound0.equals(x));
    EXPECT_TRUE(compound0.equals(compound0));
    EXPECT_FALSE(compound0.equals(compound1));
    EXPECT_FALSE(compound0.equals(compound2));
}

TEST_F(QFRFunctionality, CNOTCancellation1) {
    QuantumComputation qc(2);
    qc.x(0, 1_pc);
    qc.x(0, 1_pc);

    CircuitOptimizer::cancelCNOTs(qc);
    EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, CNOTCancellation2) {
    QuantumComputation qc(2);
    qc.swap(0, 1);
    qc.swap(1, 0);

    CircuitOptimizer::cancelCNOTs(qc);
    EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, CNOTCancellation3) {
    QuantumComputation qc(2);
    qc.swap(0, 1);
    qc.x(0, 1_pc);

    CircuitOptimizer::cancelCNOTs(qc);
    EXPECT_TRUE(qc.size() == 2U);
    auto& firstOperation = qc.front();
    EXPECT_EQ(firstOperation->getType(), qc::X);
    EXPECT_EQ(firstOperation->getTargets().front(), 0U);
    EXPECT_EQ(firstOperation->getControls().begin()->qubit, 1U);

    auto& secondOperation = qc.back();
    EXPECT_EQ(secondOperation->getType(), qc::X);
    EXPECT_EQ(secondOperation->getTargets().front(), 1U);
    EXPECT_EQ(secondOperation->getControls().begin()->qubit, 0U);
}

TEST_F(QFRFunctionality, CNOTCancellation4) {
    QuantumComputation qc(2);
    qc.x(0, 1_pc);
    qc.swap(0, 1);

    CircuitOptimizer::cancelCNOTs(qc);
    EXPECT_TRUE(qc.size() == 2U);
    auto& firstOperation = qc.front();
    EXPECT_EQ(firstOperation->getType(), qc::X);
    EXPECT_EQ(firstOperation->getTargets().front(), 1U);
    EXPECT_EQ(firstOperation->getControls().begin()->qubit, 0U);

    auto& secondOperation = qc.back();
    EXPECT_EQ(secondOperation->getType(), qc::X);
    EXPECT_EQ(secondOperation->getTargets().front(), 0U);
    EXPECT_EQ(secondOperation->getControls().begin()->qubit, 1U);
}

TEST_F(QFRFunctionality, CNOTCancellation5) {
    QuantumComputation qc(2);
    qc.x(0, 1_pc);
    qc.x(1, 0_pc);
    qc.x(0, 1_pc);

    CircuitOptimizer::cancelCNOTs(qc);
    EXPECT_TRUE(qc.size() == 1U);
    auto& firstOperation = qc.front();
    EXPECT_EQ(firstOperation->getType(), qc::SWAP);
    EXPECT_EQ(firstOperation->getTargets().front(), 0U);
    EXPECT_EQ(firstOperation->getTargets().back(), 1U);
}

TEST_F(QFRFunctionality, IndexOutOfRange) {
    QuantumComputation qc(2);
    qc::Permutation    layout{};
    layout[0]        = 0;
    layout[2]        = 1;
    qc.initialLayout = layout;
    qc.x(0);

    EXPECT_THROW(qc.x(1), QFRException);
    EXPECT_THROW(qc.x(0, dd::Control{1, dd::Control::Type::neg}), QFRException);
    EXPECT_THROW(qc.x(0, {dd::Control{2, dd::Control::Type::neg}, dd::Control{1, dd::Control::Type::neg}}), QFRException);
    EXPECT_THROW(qc.swap(0, 1), QFRException);
    EXPECT_THROW(qc.swap(0, 2, dd::Control{1, dd::Control::Type::neg}), QFRException);
    EXPECT_THROW(qc.swap(0, 2, {dd::Control{1, dd::Control::Type::neg}}), QFRException);
    EXPECT_THROW(qc.reset({0, 1, 2}), QFRException);
}

TEST_F(QFRFunctionality, ContainsLogicalQubit) {
    const QuantumComputation qc(2);
    const auto [contains0, index0] = qc.containsLogicalQubit(0);
    EXPECT_TRUE(contains0);
    EXPECT_EQ(*index0, 0);
    const auto [contains1, index1] = qc.containsLogicalQubit(1);
    EXPECT_TRUE(contains1);
    EXPECT_EQ(*index1, 1);
    const auto [contains2, index2] = qc.containsLogicalQubit(2);
    EXPECT_FALSE(contains2);
    EXPECT_FALSE(index2.has_value());
}

TEST_F(QFRFunctionality, AddAncillaryQubits) {
    QuantumComputation qc(1);
    qc.addAncillaryQubit(1, -1);
    EXPECT_EQ(qc.getNqubits(), 2);
    EXPECT_EQ(qc.getNancillae(), 1);
    ASSERT_EQ(qc.ancillary.size(), 2U);
    ASSERT_EQ(qc.garbage.size(), 2U);
    EXPECT_FALSE(qc.ancillary[0]);
    EXPECT_TRUE(qc.ancillary[1]);
    EXPECT_FALSE(qc.garbage[0]);
    EXPECT_TRUE(qc.garbage[1]);
}

TEST_F(QFRFunctionality, SingleQubitGateCount) {
    QuantumComputation qc(2);
    qc.x(0);
    qc.h(0);
    qc.x(0, 1_pc);
    qc.z(0);
    qc.measure(0, 0);

    EXPECT_EQ(qc.getNops(), 5U);
    EXPECT_EQ(qc.getNindividualOps(), 5U);
    EXPECT_EQ(qc.getNsingleQubitOps(), 3U);

    CircuitOptimizer::singleQubitGateFusion(qc);

    EXPECT_EQ(qc.getNops(), 4U);
    EXPECT_EQ(qc.getNindividualOps(), 5U);
    EXPECT_EQ(qc.getNsingleQubitOps(), 3U);
}

TEST_F(QFRFunctionality, CircuitToOperation) {
    QuantumComputation qc(1);
    EXPECT_EQ(qc.asOperation(), nullptr);
    qc.x(0);
    const auto& op = qc.asOperation();
    EXPECT_EQ(op->getType(), qc::X);
    EXPECT_EQ(op->getNqubits(), 1U);
    EXPECT_EQ(op->getNcontrols(), 0U);
    EXPECT_EQ(op->getTargets().front(), 0U);
    EXPECT_TRUE(qc.empty());
    qc.x(0);
    qc.h(0);
    const auto& op2 = qc.asOperation();
    EXPECT_EQ(op2->getType(), qc::Compound);
    EXPECT_EQ(op2->getNqubits(), 1U);
    EXPECT_TRUE(qc.empty());
}
