/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/Entanglement.hpp"

#include "gtest/gtest.h"
#include <string>

class Entanglement: public testing::TestWithParam<dd::QubitCount> {
protected:
    void TearDown() override {}
    void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(Entanglement, Entanglement,
                         testing::Range(static_cast<dd::QubitCount>(2), static_cast<dd::QubitCount>(129), 7),
                         [](const testing::TestParamInfo<Entanglement::ParamType>& info) {
                             // Generate names for test cases
                             dd::QubitCount    nqubits = info.param;
                             std::stringstream ss{};
                             ss << static_cast<std::size_t>(nqubits) << "_qubits";
                             return ss.str();
                         });

TEST_P(Entanglement, FunctionTest) {
    const dd::QubitCount nq = GetParam();

    auto                              dd = std::make_unique<dd::Package>(nq);
    std::unique_ptr<qc::Entanglement> qc;
    qc::MatrixDD                      e{};

    ASSERT_NO_THROW({ qc = std::make_unique<qc::Entanglement>(nq); });
    ASSERT_NO_THROW({ e = qc->buildFunctionality(dd); });

    ASSERT_EQ(qc->getNops(), nq);
    qc::VectorDD r = dd->multiply(e, dd->makeZeroState(nq));

    ASSERT_EQ(dd->getValueByPath(r, std::string(nq, '0')), (dd::ComplexValue{dd::SQRT2_2, 0}));
    ASSERT_EQ(dd->getValueByPath(r, std::string(nq, '1')), (dd::ComplexValue{dd::SQRT2_2, 0}));
}
