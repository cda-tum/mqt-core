/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "gtest/gtest.h"
#include <iostream>
#include <cmath>

#include "Grover.hpp"

class Grover : public testing::TestWithParam<std::tuple<unsigned short, unsigned int>> {

protected:
	void TearDown() override {
		if (!dd->isTerminal(e))
			dd->decRef(e);
		dd->garbageCollect(true);

		// number of complex table entries after clean-up should equal initial number of entries
		EXPECT_EQ(dd->cn.count, initialComplexCount);
		// number of available cache entries after clean-up should equal initial number of entries
		EXPECT_EQ(dd->cn.cacheCount, initialCacheCount);
	}

	void SetUp() override {
		dd = std::make_unique<dd::Package>();
		initialCacheCount = dd->cn.cacheCount;
		initialComplexCount = dd->cn.count;
	}

	unsigned short nqubits = 0;
	unsigned int seed = 0;
	std::unique_ptr<dd::Package> dd;
	std::unique_ptr<qc::Grover> qc;
	long initialCacheCount = 0;
	long initialComplexCount = 0;
	dd::Edge e{};
};

constexpr unsigned short GROVER_MAX_QUBITS = 14;
constexpr unsigned int GROVER_NUM_SEEDS = 10;
constexpr fp GROVER_ACCURACY = 1e-8;
constexpr fp GROVER_GOAL_PROBABILITY = 0.9;

INSTANTIATE_TEST_SUITE_P(Grover,
                         Grover,
                         testing::Combine(
		                         testing::Range((unsigned short)2,(unsigned short)(GROVER_MAX_QUBITS+1), 3),
		                         testing::Range((unsigned int)0,GROVER_NUM_SEEDS)),
                         [](const testing::TestParamInfo<Grover::ParamType>& info) {
	                         unsigned short nqubits = std::get<0>(info.param);
	                         unsigned int seed = std::get<1>(info.param);
	                         std::stringstream ss{};
	                         ss << nqubits+1;
	                         if (nqubits == 0) {
		                         ss << "_qubit_";
	                         } else {
		                         ss << "_qubits_";
	                         }
	                         ss << seed;
	                         return ss.str();});

TEST_P(Grover, Reference) {
	std::tie(nqubits, seed) = GetParam();

	// there should be no error constructing the circuit
	ASSERT_NO_THROW({qc = std::make_unique<qc::Grover>(nqubits, seed);});

	qc->printStatistics();
	unsigned long long x = dynamic_cast<qc::Grover*>(qc.get())->x;

	// there should be no error building the functionality
	ASSERT_NO_THROW({e = qc->buildFunctionality(dd);});

	// amplitudes of the searched-for entry should be 1 (up to a common factor)
	// two checks due to the ancillary qubit used (which can be either 0 or 1)
	auto c = qc->getEntry(dd, e, x, 0);
	EXPECT_NEAR(std::abs(CN::val(c.r)), 1, GROVER_ACCURACY);
	EXPECT_NEAR(CN::val(c.i), 0, GROVER_ACCURACY);
	c = qc->getEntry(dd, e, x + (1 << (nqubits + 1)), 0);
	EXPECT_NEAR(std::abs(CN::val(c.r)), 1, GROVER_ACCURACY);
	EXPECT_NEAR(CN::val(c.i), 0, GROVER_ACCURACY);

	CN::mul(c, c, e.w);
	auto prob = 2*CN::mag2(c);
	EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);
}
