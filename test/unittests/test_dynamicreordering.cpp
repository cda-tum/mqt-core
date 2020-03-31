/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "gtest/gtest.h"
#include "QuantumComputation.hpp"
#include <sstream>

class DynamicReorderingTest : public testing::TestWithParam<std::string> {

protected:
	qc::QuantumComputation qc;
	std::array<short, qc::MAX_QUBITS>       line{};
	dd::Edge                                e{}, in{};
	std::unique_ptr<dd::Package>            dd;

	std::string circuit_dir = "./circuits/";

	void SetUp() override {
		dd = std::make_unique<dd::Package>();
		line.fill(qc::LINE_DEFAULT);
		qc.import(circuit_dir + GetParam());
	}

	void TearDown() override {

	}

};

INSTANTIATE_TEST_SUITE_P(SomeCircuits, DynamicReorderingTest, testing::Values("bell.qasm"),
		[](const testing::TestParamInfo<DynamicReorderingTest::ParamType>& info) {
			auto s = info.param;
			std::replace( s.begin(), s.end(), '.', '_');
			return s;
		});

TEST_P(DynamicReorderingTest, simulation) {
	in = dd->makeZeroState(qc.getNqubits());
	e = qc.simulate(in, dd, dd::Sifting);
	std::stringstream ss{};
	ss << GetParam() << "_vector_sifting.dot";
	dd->export2Dot(e, ss.str().c_str(), true);
}

TEST_P(DynamicReorderingTest, construction) {
	e = qc.buildFunctionality(dd, dd::Sifting);
	std::stringstream ss{};
	ss << GetParam() << "_matrix_sifting.dot";
	dd->export2Dot(e, ss.str().c_str(), false);
}
