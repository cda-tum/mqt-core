/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/RandomCliffordCircuit.hpp"

#include "gtest/gtest.h"
#include <string>

class RandomClifford: public testing::TestWithParam<unsigned short> {
protected:
    void TearDown() override {}
    void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(RandomClifford, RandomClifford,
                         testing::Range((unsigned short)1, (unsigned short)9),
                         [](const testing::TestParamInfo<RandomClifford::ParamType>& info) {
                             // Generate names for test cases
                             unsigned short    nqubits = info.param;
                             std::stringstream ss{};
                             ss << nqubits << "_qubits";
                             return ss.str();
                         });

TEST_P(RandomClifford, simulate) {
    const unsigned short nq = GetParam();

    auto dd = std::make_unique<dd::Package>();
    auto qc = qc::RandomCliffordCircuit(nq, nq * nq);
    auto in = dd->makeZeroState(nq);

    std::cout << qc << std::endl;
    ASSERT_NO_THROW({ qc.simulate(in, dd); });
    qc.printStatistics(std::cout);
}

TEST_P(RandomClifford, buildFunctionality) {
    const unsigned short nq = GetParam();

    auto dd = std::make_unique<dd::Package>();
    auto qc = qc::RandomCliffordCircuit(nq, nq * nq);
    std::cout << qc << std::endl;
    ASSERT_NO_THROW({ qc.buildFunctionality(dd); });
    qc.printStatistics(std::cout);
}
