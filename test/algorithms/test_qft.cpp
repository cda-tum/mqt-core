/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "gtest/gtest.h"
#include <iostream>
#include <cmath>

#include "algorithms/QFT.hpp"

class QFT : public testing::TestWithParam<unsigned short> {

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
	std::unique_ptr<dd::Package> dd;
	std::unique_ptr<qc::QFT> qc;
	long initialCacheCount = 0;
	long initialComplexCount = 0;
	dd::Edge e{};
};

/// Findings from the QFT Benchmarks:
/// The DDpackage has to be able to represent all 2^n different amplitudes in order to produce correct results
/// The smallest entry seems to be closely related to '1-cos(pi/2^(n-1))'
/// The following CN::TOLERANCE values suffice up until a certain number of qubits:
/// 	10e-10	..	18 qubits
///		10e-11	..	20 qubits
///		10e-12	..	22 qubits
///		10e-13	..	23 qubits
/// The accuracy of double floating points allows for a minimal CN::TOLERANCE value of 10e-15
///	Utilizing more qubits requires the use of fp=long double
constexpr unsigned short QFT_MAX_QUBITS = 20;

INSTANTIATE_TEST_SUITE_P(QFT, QFT,
		testing::Range((unsigned short)0,(unsigned short)(QFT_MAX_QUBITS+1), 3),
		[](const testing::TestParamInfo<QFT::ParamType>& info) {
			unsigned short nqubits = info.param;
			std::stringstream ss{};
			ss << nqubits;
			if (nqubits == 1) {
				ss << "_qubit";
			} else {
				ss << "_qubits";
			}
			return ss.str();});

TEST_P(QFT, Reference) {
	nqubits = GetParam();

	// there should be no error constructing the circuit
	ASSERT_NO_THROW({qc = std::make_unique<qc::QFT>(nqubits);});

	// there should be no error building the functionality
	ASSERT_NO_THROW({e = qc->buildFunctionality(dd);});

	qc->printStatistics();

	// QFT DD should consist of 2^n nodes
	ASSERT_EQ(dd->size(e), std::pow(2, nqubits));

	// Force garbage collection of compute table and complex table
	dd->garbageCollect(true);

	// the final DD should store all 2^n different amplitudes
	// since only positive real values are stored in the complex table
	// this number has to be divided by 4
	// the (+3) accounts for the fact that the table is pre-filled with some values {0,0.5,sqrt(0.5)}
	ASSERT_EQ(dd->cn.count, (unsigned int)(std::ceil(std::pow(2,nqubits)/4))+3);

	// top edge weight should equal sqrt(0.5)^n
	EXPECT_NEAR(CN::val(e.w.r), std::pow(1.L/std::sqrt(2.L), nqubits), CN::TOLERANCE);

	// first row and first column should consist only of 1's
	for (int i = 0; i < std::pow(2, nqubits); ++i) {
		auto c = qc->getEntry(dd, e, 0, i);
		EXPECT_NEAR(CN::val(c.r), 1, CN::TOLERANCE);
		EXPECT_NEAR(CN::val(c.i), 0, CN::TOLERANCE);
		c = qc->getEntry(dd, e, i, 0);
		EXPECT_NEAR(CN::val(c.r), 1, CN::TOLERANCE);
		EXPECT_NEAR(CN::val(c.i), 0, CN::TOLERANCE);
	}
}

TEST_P(QFT, ReferenceSim) {
	nqubits = GetParam();

	// there should be no error constructing the circuit
	ASSERT_NO_THROW({qc = std::make_unique<qc::QFT>(nqubits);});

	// there should be no error building the functionality
	ASSERT_NO_THROW({
		dd::Edge in = dd->makeZeroState(nqubits);
        e = qc->simulate(in, dd);
	});
	qc->printStatistics();

	// QFT DD |0...0> sim should consist of n nodes
	ASSERT_EQ(dd->size(e), nqubits+1);

	// Force garbage collection of compute table and complex table
	dd->garbageCollect(true);

	// top edge weight should equal sqrt(0.5)^n
	EXPECT_NEAR(CN::val(e.w.r), 1, CN::TOLERANCE);
	EXPECT_NEAR(CN::val(e.w.i), 0, CN::TOLERANCE);

	// first column should consist only of 1's
	for (int i = 0; i < std::pow(2, nqubits); ++i) {
		auto c = qc->getEntry(dd, e, i, 0);
		EXPECT_NEAR(CN::val(c.r), std::pow(1.L/std::sqrt(2.L), nqubits), CN::TOLERANCE);
		EXPECT_NEAR(CN::val(c.i), 0, CN::TOLERANCE);
	}
}
