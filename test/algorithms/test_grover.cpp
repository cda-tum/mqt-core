/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "algorithms/Grover.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"

#include "gtest/gtest.h"
#include <cmath>
#include <iostream>

class Grover: public testing::TestWithParam<std::tuple<dd::QubitCount, std::size_t>> {
protected:
    void TearDown() override {
        if (!sim.isTerminal())
            dd->decRef(sim);
        if (!func.isTerminal())
            dd->decRef(func);
        dd->garbageCollect(true);

        // number of complex table entries after clean-up should equal initial number of entries
        EXPECT_EQ(dd->cn.complexTable.getCount(), initialComplexCount);
        // if (dd->cn.complexTable.getCount() != initialComplexCount) {
        //     dd->cn.complexTable.print();
        // }

        // number of available cache entries after clean-up should equal initial number of entries
        EXPECT_EQ(dd->cn.complexCache.getCount(), initialCacheCount);
    }

    void SetUp() override {
        std::tie(nqubits, seed) = GetParam();
        dd                      = std::make_unique<dd::Package<>>(nqubits + 1);
        initialCacheCount       = dd->cn.complexCache.getCount();
        initialComplexCount     = dd->cn.complexTable.getCount();
    }

    dd::QubitCount                 nqubits = 0;
    std::size_t                    seed    = 0;
    std::unique_ptr<dd::Package<>> dd;
    std::unique_ptr<qc::Grover>    qc;
    std::size_t                    initialCacheCount   = 0;
    std::size_t                    initialComplexCount = 0;
    qc::VectorDD                   sim{};
    qc::MatrixDD                   func{};
};

constexpr dd::QubitCount GROVER_MAX_QUBITS       = 18;
constexpr std::size_t    GROVER_NUM_SEEDS        = 5;
constexpr dd::fp         GROVER_ACCURACY         = 1e-2;
constexpr dd::fp         GROVER_GOAL_PROBABILITY = 0.9;

INSTANTIATE_TEST_SUITE_P(Grover,
                         Grover,
                         testing::Combine(
                                 testing::Range(static_cast<dd::QubitCount>(2), static_cast<dd::QubitCount>(GROVER_MAX_QUBITS + 1), 3),
                                 testing::Range(static_cast<std::size_t>(0), GROVER_NUM_SEEDS)),
                         [](const testing::TestParamInfo<Grover::ParamType>& info) {
	                         dd::QubitCount nqubits = std::get<0>(info.param);
	                         std::size_t seed = std::get<1>(info.param);
	                         std::stringstream ss{};
	                         ss << static_cast<std::size_t>(nqubits+1);
	                         if (nqubits == 0) {
		                         ss << "_qubit_";
	                         } else {
		                         ss << "_qubits_";
	                         }
	                         ss << seed;
	                         return ss.str(); });

TEST_P(Grover, Functionality) {
    // there should be no error constructing the circuit
    ASSERT_NO_THROW({ qc = std::make_unique<qc::Grover>(nqubits, seed); });

    qc->printStatistics(std::cout);
    auto x = '1' + dynamic_cast<qc::Grover*>(qc.get())->expected;
    std::reverse(x.begin(), x.end());
    std::replace(x.begin(), x.end(), '1', '2');

    // there should be no error building the functionality
    ASSERT_NO_THROW({ func = buildFunctionality(qc.get(), dd); });

    // amplitude of the searched-for entry should be 1
    auto c = dd->getValueByPath(func, x);
    EXPECT_NEAR(std::abs(c.r), 1, GROVER_ACCURACY);
    EXPECT_NEAR(std::abs(c.i), 0, GROVER_ACCURACY);
    auto prob = c.r * c.r + c.i * c.i;
    EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);
}

TEST_P(Grover, FunctionalityRecursive) {
    // there should be no error constructing the circuit
    ASSERT_NO_THROW({ qc = std::make_unique<qc::Grover>(nqubits, seed); });

    qc->printStatistics(std::cout);
    auto x = '1' + dynamic_cast<qc::Grover*>(qc.get())->expected;
    std::reverse(x.begin(), x.end());
    std::replace(x.begin(), x.end(), '1', '2');

    // there should be no error building the functionality
    ASSERT_NO_THROW({ func = buildFunctionalityRecursive(qc.get(), dd); });

    // amplitude of the searched-for entry should be 1
    auto c = dd->getValueByPath(func, x);
    EXPECT_NEAR(std::abs(c.r), 1, GROVER_ACCURACY);
    EXPECT_NEAR(std::abs(c.i), 0, GROVER_ACCURACY);
    auto prob = c.r * c.r + c.i * c.i;
    EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);
}

TEST_P(Grover, Simulation) {
    // there should be no error constructing the circuit
    ASSERT_NO_THROW({ qc = std::make_unique<qc::Grover>(nqubits, seed); });

    qc->printStatistics(std::cout);
    auto in = dd->makeZeroState(nqubits + 1);
    // there should be no error simulating the circuit
    std::size_t shots        = 1024;
    auto        measurements = simulate(qc.get(), in, dd, shots);

    for (const auto& [state, count]: measurements) {
        std::cout << state << ": " << count << std::endl;
    }

    auto correctShots = measurements[qc->expected];
    auto probability  = static_cast<double>(correctShots) / static_cast<double>(shots);

    EXPECT_GE(probability, GROVER_GOAL_PROBABILITY);
}
