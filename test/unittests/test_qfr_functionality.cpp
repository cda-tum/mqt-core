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
	CircuitOptimizer::swapGateFusion(qc);
	ASSERT_NO_THROW({
		auto op = dynamic_cast<StandardOperation*>((qc.begin()->get()));
        EXPECT_EQ(op->getGate(), SWAP);
        EXPECT_EQ(op->getTargets().at(0), 0);
        EXPECT_EQ(op->getTargets().at(1), 1);
	});
}

TEST_F(QFRFunctionality, replace_cx_to_swap_at_end) {
	unsigned short nqubits = 2;
	QuantumComputation qc(nqubits);
	qc.emplace_back<StandardOperation>(nqubits, Control(0), 1, X);
	qc.emplace_back<StandardOperation>(nqubits, Control(1), 0, X);
	CircuitOptimizer::swapGateFusion(qc);
	auto it = qc.begin();
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getGate(), SWAP);
		                EXPECT_EQ(op->getTargets().at(0), 0);
		                EXPECT_EQ(op->getTargets().at(1), 1);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getGate(), X);
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
	CircuitOptimizer::swapGateFusion(qc);
	auto it = qc.begin();
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getGate(), SWAP);
		                EXPECT_EQ(op->getTargets().at(0), 0);
		                EXPECT_EQ(op->getTargets().at(1), 1);
	                });
	++it;
	ASSERT_NO_THROW({
		                auto op = dynamic_cast<StandardOperation*>(it->get());
		                EXPECT_EQ(op->getGate(), X);
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
	EXPECT_TRUE(dd->equals(e.p->e[0], dd->makeIdent(0,nqubits-1)));
	EXPECT_TRUE(dd->equals(e.p->e[1], dd->DDzero));
	EXPECT_TRUE(dd->equals(e.p->e[2], dd->DDzero));
	EXPECT_TRUE(dd->equals(e.p->e[3], dd->DDzero));
	auto f = dd->makeIdent(0, (short)nqubits);
	dd->incRef(f);
	qc.reduceAncillae(f, dd);
	qc.reduceGarbage(f, dd);
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
	qc.printStatistics();
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
