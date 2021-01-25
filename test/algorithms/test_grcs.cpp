/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/GoogleRandomCircuitSampling.hpp"
#include "gtest/gtest.h"

class GRCS : public testing::TestWithParam<unsigned short> {

protected:
	void TearDown() override {}
	void SetUp() override {}
};

TEST_F(GRCS, import) {
	auto qc_bris = qc::GoogleRandomCircuitSampling("./circuits/grcs/bris_4_40_9_v2.txt");
	qc_bris.printStatistics(std::cout);
	std::cout << qc_bris << std::endl;

	auto qc_inst = qc::GoogleRandomCircuitSampling("./circuits/grcs/inst_4x4_80_9_v2.txt");
	qc_inst.printStatistics(std::cout);
	std::cout << qc_inst << std::endl;
}

TEST_F(GRCS, simulate) {
	auto qc_bris = qc::GoogleRandomCircuitSampling("./circuits/grcs/bris_4_40_9_v2.txt");

	auto dd = std::make_unique<dd::Package>();
    auto in = dd->makeZeroState(qc_bris.getNqubits());
	ASSERT_NO_THROW({
        qc_bris.simulate(in, dd, 4);
    });
	std::cout << qc_bris << std::endl;
	qc_bris.printStatistics(std::cout);
}

TEST_F(GRCS, buildFunctionality) {
	auto qc_bris = qc::GoogleRandomCircuitSampling("./circuits/grcs/bris_4_40_9_v2.txt");

	auto dd = std::make_unique<dd::Package>();
	ASSERT_NO_THROW({
		               qc_bris.buildFunctionality(dd, 4);
	                });
	std::cout << qc_bris << std::endl;
	qc_bris.printStatistics(std::cout);
}
