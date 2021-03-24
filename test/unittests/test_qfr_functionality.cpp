/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "gtest/gtest.h"
#include <random>

#include "QuantumComputation.hpp"
#include "CircuitOptimizer.hpp"

using namespace qc;

class QFRFunctionality : public testing::TestWithParam<unsigned short> {
protected:
	void TearDown() override {

	}

	void SetUp() override {
		dd = std::make_unique<dd::Package>();
		line.fill(qc::LINE_DEFAULT);

		std::array<std::mt19937_64::result_type , std::mt19937_64::state_size> random_data{};
		std::random_device rd;
		std::generate(begin(random_data), end(random_data), [&](){return rd();});
		std::seed_seq seeds(begin(random_data), end(random_data));
		mt.seed(seeds);
		dist = std::uniform_real_distribution<fp> (0.0, 2 * qc::PI);
	}

	std::array<short, qc::MAX_QUBITS>       line{};
	std::unique_ptr<dd::Package>            dd;
	std::mt19937_64                         mt;
	std::uniform_real_distribution<fp>      dist;
};

TEST_F(QFRFunctionality, fuse_cx_to_swap) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, Control(0), 1, X);
	qc.emplace_back<StandardOperation>(nqubits, Control(1), 0, X);
	qc.emplace_back<StandardOperation>(nqubits, Control(0), 1, X);
	CircuitOptimizer::swapReconstruction(qc);
	ASSERT_NO_THROW({
		auto op = dynamic_cast<StandardOperation*>((qc.begin()->get()));
        EXPECT_EQ(op->getType(), SWAP);
        EXPECT_EQ(op->getTargets().at(0), 0);
        EXPECT_EQ(op->getTargets().at(1), 1);
	});
}

TEST_F(QFRFunctionality, replace_cx_to_swap_at_end) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, Control(0), 1, X);
	qc.emplace_back<StandardOperation>(nqubits, Control(1), 0, X);
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
		                EXPECT_EQ(op->getType(), X);
		                EXPECT_EQ(op->getControls().at(0).qubit, 0);
		                EXPECT_EQ(op->getTargets().at(0), 1);
	                });
}

TEST_F(QFRFunctionality, replace_cx_to_swap) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, Control(0), 1, X);
	qc.emplace_back<StandardOperation>(nqubits, Control(1), 0, X);
	qc.emplace_back<StandardOperation>(nqubits, 0, H);
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
		                EXPECT_EQ(op->getType(), X);
		                EXPECT_EQ(op->getControls().at(0).qubit, 0);
		                EXPECT_EQ(op->getTargets().at(0), 1);
	                });
}

TEST_F(QFRFunctionality, remove_trailing_idle_qubits) {
	unsigned short nqubits = 4;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, X);
	qc.emplace_back<StandardOperation>(nqubits, 2, X);
	std::cout << qc;
	qc::QuantumComputation::printPermutationMap(qc.outputPermutation);
	qc.printRegisters();

	qc.outputPermutation.erase(1);
	qc.outputPermutation.erase(3);

	qc.stripIdleQubits();
	EXPECT_EQ(qc.getNqubits(), 2);
	std::cout << qc;
	qc::QuantumComputation::printPermutationMap(qc.outputPermutation);
	qc.printRegisters();

	qc.pop_back();
	qc.outputPermutation.erase(2);
	std::cout << qc;
	qc::QuantumComputation::printPermutationMap(qc.outputPermutation);
	qc.printRegisters();

	qc.stripIdleQubits();
	EXPECT_EQ(qc.getNqubits(), 1);
}

TEST_F(QFRFunctionality, ancillary_qubit_at_end) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, X);
	qc.addAncillaryRegister(1);
	EXPECT_EQ(qc.getNancillae(), 1);
	EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
	EXPECT_EQ(qc.getNqubits(), 3);
	qc.emplace_back<StandardOperation>(nqubits, 2, X);
	auto e = qc.createInitialMatrix(dd);
	EXPECT_TRUE(dd->equals(e.p->e[0], dd->makeIdent(nqubits)));
	EXPECT_TRUE(dd->equals(e.p->e[1], dd->DDzero));
	EXPECT_TRUE(dd->equals(e.p->e[2], dd->DDzero));
	EXPECT_TRUE(dd->equals(e.p->e[3], dd->DDzero));
	auto f = dd->makeIdent(nqubits+1);
	dd->incRef(f);
	f = qc.reduceAncillae(f, dd);
	f = qc.reduceGarbage(f, dd);
	EXPECT_TRUE(dd->equals(e, f));
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
	EXPECT_EQ(qc.getNqubits(), nqubits+1);
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
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, X);
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
	unsigned short nqubits = 3;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, X);
	auto p = qc.removeQubit(1);
	EXPECT_EQ(p.first, 1);
	EXPECT_EQ(p.second, 1);
	EXPECT_EQ(qc.getNancillae(), 0);
	EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 2);
	EXPECT_EQ(qc.getNqubits(), 2);
	qc.printRegisters();
}

TEST_F(QFRFunctionality, FuseTwoSingleQubitGates) {
	unsigned short nqubits = 1;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, X);
	qc.emplace_back<StandardOperation>(nqubits, 0, H);

	qc.print(std::cout);
	dd::Edge e = qc.buildFunctionality(dd);
	CircuitOptimizer::singleQubitGateFusion(qc);
	dd::Edge f = qc.buildFunctionality(dd);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 1);
	EXPECT_TRUE(dd::Package::equals(e, f));
}

TEST_F(QFRFunctionality, FuseThreeSingleQubitGates) {
	unsigned short nqubits = 1;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, X);
	qc.emplace_back<StandardOperation>(nqubits, 0, H);
	qc.emplace_back<StandardOperation>(nqubits, 0, Y);

	dd::Edge e = qc.buildFunctionality(dd);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::singleQubitGateFusion(qc);
	dd::Edge f = qc.buildFunctionality(dd);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 1);
	EXPECT_TRUE(dd::Package::equals(e, f));
}

TEST_F(QFRFunctionality, FuseNoSingleQubitGates) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, H);
	qc.emplace_back<StandardOperation>(nqubits, qc::Control(0), 1, X);
	qc.emplace_back<StandardOperation>(nqubits, 0, Y);
	dd::Edge e = qc.buildFunctionality(dd);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::singleQubitGateFusion(qc);
	dd::Edge f = qc.buildFunctionality(dd);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 3);
	EXPECT_TRUE(dd::Package::equals(e, f));
}

TEST_F(QFRFunctionality, FuseSingleQubitGatesAcrossOtherGates) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, H);
	qc.emplace_back<StandardOperation>(nqubits, 1, Z);
	qc.emplace_back<StandardOperation>(nqubits, 0, Y);
	auto e = qc.buildFunctionality(dd);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::singleQubitGateFusion(qc);
	auto f = qc.buildFunctionality(dd);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 2);
	EXPECT_TRUE(dd::Package::equals(e, f));
}

TEST_F(QFRFunctionality, StripIdleAndDump) {
	std::stringstream ss{};
	auto testfile =
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
	unsigned short nqubits = 1;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, X);
	qc.emplace_back<StandardOperation>(nqubits, 0, I);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::singleQubitGateFusion(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 1);
	EXPECT_TRUE(qc.begin()->get()->isStandardOperation());
}

TEST_F(QFRFunctionality, eliminateCompoundOperation) {
	unsigned short nqubits = 1;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, I);
	qc.emplace_back<StandardOperation>(nqubits, 0, I);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::singleQubitGateFusion(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 0);
	EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, eliminateInverseInCompoundOperation) {
	unsigned short nqubits = 1;
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
	unsigned short nqubits = 1;
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
	unsigned short nqubits = 1;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, Z);
	qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{0}, std::vector<unsigned short>{0});
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 1);
	EXPECT_EQ(qc.begin()->get()->getType(), qc::Measure);
}

TEST_F(QFRFunctionality, removeDiagonalCompoundOpBeforeMeasure) {
	unsigned short nqubits = 1;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, Z);
	qc.emplace_back<StandardOperation>(nqubits, 0, T);
	qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{0}, std::vector<unsigned short>{0});
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
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, qc::Control(0), 1, Z);
	qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{0,1}, std::vector<unsigned short>{0,1});
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 1);
	EXPECT_EQ(qc.begin()->get()->getType(), qc::Measure);
}

TEST_F(QFRFunctionality, leaveGateBeforeMeasure) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, qc::Control(0), 1, Z);
	qc.emplace_back<StandardOperation>(nqubits, 0, X);
	qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{0,1}, std::vector<unsigned short>{0,1});
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 3);
}

TEST_F(QFRFunctionality, removeComplexGateBeforeMeasure) {
	unsigned short nqubits = 4;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, qc::Control(0), 1, Z);
	qc.emplace_back<StandardOperation>(nqubits, 0, X);
	qc.emplace_back<StandardOperation>(nqubits, qc::Control(1), 2, Z);
	qc.emplace_back<StandardOperation>(nqubits, qc::Control(0), 1, Z);
	qc.emplace_back<StandardOperation>(nqubits, 0, Z);
	qc.emplace_back<StandardOperation>(nqubits, qc::Control(1), 2, Z);
	qc.emplace_back<StandardOperation>(nqubits, 3, X);
	qc.emplace_back<StandardOperation>(nqubits, 3, T);
	qc.emplace_back<StandardOperation>(nqubits, std::vector<qc::Control>{qc::Control(0), qc::Control(1), qc::Control(2)}, 3, Z);

	qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{0,1,2,3}, std::vector<unsigned short>{0,1,2,3});
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 4);
}

TEST_F(QFRFunctionality, removeSimpleCompoundOpBeforeMeasure) {
	unsigned short nqubits = 1;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, X);
	qc.emplace_back<StandardOperation>(nqubits, 0, T);
	qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{0}, std::vector<unsigned short>{0});
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::singleQubitGateFusion(qc);
	CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 2);
}

TEST_F(QFRFunctionality, removePartOfCompoundOpBeforeMeasure) {
	unsigned short nqubits = 1;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, T);
	qc.emplace_back<StandardOperation>(nqubits, 0, X);
	qc.emplace_back<StandardOperation>(nqubits, 0, T);
	qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{0}, std::vector<unsigned short>{0});
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::singleQubitGateFusion(qc);
	CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	EXPECT_EQ(qc.getNops(), 2);
}

TEST_F(QFRFunctionality, decomposeSWAPsUndirectedArchitecture) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, std::vector<qc::Control>{}, 0,1, SWAP);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::decomposeSWAP(qc, false);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	auto it = qc.begin();
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), X);
		                EXPECT_EQ(op->getControls().at(0).qubit, 0);
		                EXPECT_EQ(op->getTargets().at(0), 1);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), X);
		                EXPECT_EQ(op->getControls().at(0).qubit, 1);
		                EXPECT_EQ(op->getTargets().at(0), 0);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), X);
		                EXPECT_EQ(op->getControls().at(0).qubit, 0);
		                EXPECT_EQ(op->getTargets().at(0), 1);
	                });
}
TEST_F(QFRFunctionality, decomposeSWAPsDirectedArchitecture) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, std::vector<qc::Control>{}, 0,1, SWAP);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::decomposeSWAP(qc, true);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	auto it = qc.begin();
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), X);
		                EXPECT_EQ(op->getControls().at(0).qubit, 0);
		                EXPECT_EQ(op->getTargets().at(0), 1);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), H);
		                EXPECT_EQ(op->getTargets().at(0), 1);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), H);
		                EXPECT_EQ(op->getTargets().at(0), 0);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), X);
		                EXPECT_EQ(op->getControls().at(0).qubit, 0);
		                EXPECT_EQ(op->getTargets().at(0), 1);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), H);
		                EXPECT_EQ(op->getTargets().at(0), 1);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), H);
		                EXPECT_EQ(op->getTargets().at(0), 0);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), X);
		                EXPECT_EQ(op->getControls().at(0).qubit, 0);
		                EXPECT_EQ(op->getTargets().at(0), 1);
	                }); 
}

TEST_F(QFRFunctionality, decomposeSWAPsCompound) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X); //We need something to replace with compound
	auto gate = std::make_unique<qc::CompoundOperation>(nqubits);
	gate->emplace_back<qc::StandardOperation>(nqubits, std::vector<qc::Control>{}, 0,1, qc::SWAP);
	gate->emplace_back<qc::StandardOperation>(nqubits, std::vector<qc::Control>{}, 0,1, qc::SWAP);
	gate->emplace_back<qc::StandardOperation>(nqubits, std::vector<qc::Control>{}, 0,1, qc::SWAP);
	(*qc.begin()) = std::move(gate);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	qc::CircuitOptimizer::decomposeSWAP(qc, false);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	auto it = qc.begin();
	EXPECT_EQ(it->get()->isCompoundOperation(), true);

	EXPECT_EQ(dynamic_cast<qc::CompoundOperation *>(it->get())->size(), 9);
}
TEST_F(QFRFunctionality, decomposeSWAPsCompoundDirected) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X); //We need something to replace with compound
	auto gate = std::make_unique<qc::CompoundOperation>(nqubits);
	gate->emplace_back<qc::StandardOperation>(nqubits, std::vector<qc::Control>{}, 0,1, qc::SWAP);
	gate->emplace_back<qc::StandardOperation>(nqubits, std::vector<qc::Control>{}, 0,1, qc::SWAP);
	gate->emplace_back<qc::StandardOperation>(nqubits, std::vector<qc::Control>{}, 0,1, qc::SWAP);
	(*qc.begin()) = std::move(gate);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	qc::CircuitOptimizer::decomposeSWAP(qc, true);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	auto it = qc.begin();
	EXPECT_EQ(it->get()->isCompoundOperation(), true);

	EXPECT_EQ(dynamic_cast<qc::CompoundOperation *>(it->get())->size(), 21);
}

TEST_F(QFRFunctionality, removeFinalMeasurements) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, 0, H);
	qc.emplace_back<StandardOperation>(nqubits, 1, H);
	qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{0}, std::vector<unsigned short>{0});
	qc.emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{1}, std::vector<unsigned short>{1});
	qc.emplace_back<StandardOperation>(nqubits, 1, H);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::removeFinalMeasurements(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	auto it = qc.begin();
	++it;++it; //skip first two H
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<NonUnitaryOperation*>(it->get());
		                EXPECT_EQ(op->getType(), Measure);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), H);
						EXPECT_EQ(op->getTargets().at(0), 1);
	                });
}

TEST_F(QFRFunctionality, removeFinalMeasurementsCompound) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X); //We need something to replace with compound
	auto gate = std::make_unique<qc::CompoundOperation>(nqubits);
	gate->emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{0}, std::vector<unsigned short>{0});
	gate->emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{1}, std::vector<unsigned short>{1});
	gate->emplace_back<StandardOperation>(nqubits, 1, H);
	(*qc.begin()) = std::move(gate);
	qc.emplace_back<StandardOperation>(nqubits, 1, H);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::removeFinalMeasurements(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	auto it = qc.begin();
	EXPECT_EQ(it->get()->isCompoundOperation(), true);

	EXPECT_EQ(dynamic_cast<qc::CompoundOperation *>(it->get())->size(), 2);
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), H);
						EXPECT_EQ(op->getTargets().at(0), 1);
	                });
}

TEST_F(QFRFunctionality, removeFinalMeasurementsCompoundDegraded) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X); //We need something to replace with compound
	auto gate = std::make_unique<qc::CompoundOperation>(nqubits);
	gate->emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{0}, std::vector<unsigned short>{0});
	gate->emplace_back<StandardOperation>(nqubits, 1, H);
	(*qc.begin()) = std::move(gate);
	qc.emplace_back<StandardOperation>(nqubits, 1, H);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::removeFinalMeasurements(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	auto it = qc.begin();
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), H);
						EXPECT_EQ(op->getTargets().at(0), 1);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), H);
						EXPECT_EQ(op->getTargets().at(0), 1);
	                });
}

TEST_F(QFRFunctionality, removeFinalMeasurementsCompoundEmpty) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X); //We need something to replace with compound
	auto gate = std::make_unique<qc::CompoundOperation>(nqubits);
	gate->emplace_back<NonUnitaryOperation>(nqubits, std::vector<unsigned short>{0}, std::vector<unsigned short>{0});
	(*qc.begin()) = std::move(gate);
	qc.emplace_back<StandardOperation>(nqubits, 1, H);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	CircuitOptimizer::removeFinalMeasurements(qc);
	std::cout << "-----------------------------" << std::endl;
	qc.print(std::cout);
	auto it = qc.begin();
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getType(), H);
						EXPECT_EQ(op->getTargets().at(0), 1);
	                });
}
