/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "CircuitOptimizer.hpp"
#include "QuantumComputation.hpp"

#include "gtest/gtest.h"
#include <random>

using namespace qc;
using namespace dd;

class QFRFunctionality: public testing::TestWithParam<dd::QubitCount> {
protected:
    void TearDown() override {
    }

    void SetUp() override {
        dd = std::make_unique<dd::Package>(5);

        std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> random_data{};
        std::random_device                                                    rd;
        std::generate(begin(random_data), end(random_data), [&]() { return rd(); });
        std::seed_seq seeds(begin(random_data), end(random_data));
        mt.seed(seeds);
        dist = std::uniform_real_distribution<dd::fp>(0.0, 2 * dd::PI);
    }

    std::unique_ptr<dd::Package>           dd;
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
    auto e = qc.createInitialMatrix(dd);
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
    dd::Edge e = qc.buildFunctionality(dd);
    CircuitOptimizer::singleQubitGateFusion(qc);
    dd::Edge f = qc.buildFunctionality(dd);
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

    dd::Edge e = qc.buildFunctionality(dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    dd::Edge f = qc.buildFunctionality(dd);
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
    dd::Edge e = qc.buildFunctionality(dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    dd::Edge f = qc.buildFunctionality(dd);
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
    auto e = qc.buildFunctionality(dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    auto f = qc.buildFunctionality(dd);
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
    ASSERT_EQ(qc.getNops(), 3);
    ASSERT_EQ(qc.getNindividualOps(), 7);
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

TEST_F(QFRFunctionality, basicTensorDumpTest) {
    QuantumComputation qc(2);
    qc.h(1);
    qc.x(0, 1_pc);

    std::stringstream ss{};
    qc.dump(ss, qc::Tensor);

    auto reference = "{\"tensors\": [\n"
                     "[[\"H   \", \"Q1\", \"GATE0\"], [\"q1_0\", \"q1_1\"], [2, 2], [[0.70710678118654757, 0], [0.70710678118654757, 0], [0.70710678118654757, 0], [-0.70710678118654757, 0]]],\n"
                     "[[\"X   \", \"Q1\", \"Q0\", \"GATE1\"], [\"q1_1\", \"q0_0\", \"q1_2\", \"q0_1\"], [4, 4], [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [1, 0], [0, 0]]]\n"
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
                     "[[\"X   \", \"Q1\", \"Q0\", \"GATE1\"], [\"q1_1\", \"q0_0\", \"q1_2\", \"q0_1\"], [4, 4], [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [1, 0], [0, 0]]]\n"
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
