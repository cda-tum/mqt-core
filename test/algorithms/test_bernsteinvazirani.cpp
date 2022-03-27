/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "CircuitOptimizer.hpp"
#include "algorithms/BernsteinVazirani.hpp"
#include "dd/Simulation.hpp"

#include "gtest/gtest.h"

class BernsteinVazirani: public testing::TestWithParam<std::uint_least64_t> {
protected:
    void TearDown() override {}
    void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(BernsteinVazirani, BernsteinVazirani,
                         testing::Values(0ull,                                   // Zero-Value
                                         3ull, 63ull, 170ull,                    // 0-bit < hInt <= 8-bit
                                         819ull, 4032ull, 33153ull,              // 8-bit < hInt <= 16-bit
                                         87381ull, 16777215ull, 1234567891011ull // 16-bit < hInt <= 32-bit
                                         ),
                         [](const testing::TestParamInfo<BernsteinVazirani::ParamType>& info) {
                             // Generate names for test cases
                             std::uint_least64_t s = info.param;
                             std::stringstream   ss{};
                             ss << "bv_" << s;
                             return ss.str();
                         });

TEST_P(BernsteinVazirani, FunctionTest) {
    // get hidden bitstring
    auto s = qc::BitString(GetParam());

    // construct Bernstein Vazirani circuit
    auto qc = std::make_unique<qc::BernsteinVazirani>(s);
    qc->printStatistics(std::cout);

    // simulate the circuit
    auto        dd           = std::make_unique<dd::Package<>>(qc->getNqubits());
    std::size_t shots        = 1024;
    auto        measurements = simulate(qc.get(), dd->makeZeroState(qc->getNqubits()), dd, shots);

    for (const auto& [state, count]: measurements) {
        std::cout << state << ": " << count << std::endl;
    }

    // expect to obtain the hidden bitstring with certainty
    EXPECT_EQ(measurements[qc->expected], shots);
}

TEST_P(BernsteinVazirani, FunctionTestDynamic) {
    // get hidden bitstring
    auto s = qc::BitString(GetParam());

    // construct Bernstein Vazirani circuit
    auto qc = std::make_unique<qc::BernsteinVazirani>(s, true);
    qc->printStatistics(std::cout);

    // simulate the circuit
    auto        dd           = std::make_unique<dd::Package<>>(qc->getNqubits());
    std::size_t shots        = 1024;
    auto        measurements = simulate(qc.get(), dd->makeZeroState(qc->getNqubits()), dd, shots);

    for (const auto& [state, count]: measurements) {
        std::cout << state << ": " << count << std::endl;
    }

    // expect to obtain the hidden bitstring with certainty
    EXPECT_EQ(measurements[qc->expected], shots);
}

TEST_F(BernsteinVazirani, LargeCircuit) {
    dd::QubitCount nq = 127;
    auto           qc = std::make_unique<qc::BernsteinVazirani>(nq);
    qc->printStatistics(std::cout);

    // simulate the circuit
    auto        dd           = std::make_unique<dd::Package<>>(qc->getNqubits());
    std::size_t shots        = 1024;
    auto        measurements = simulate(qc.get(), dd->makeZeroState(qc->getNqubits()), dd, shots);

    for (const auto& [state, count]: measurements) {
        std::cout << state << ": " << count << std::endl;
    }

    // expect to obtain the hidden bitstring with certainty
    EXPECT_EQ(measurements[qc->expected], shots);
}

TEST_F(BernsteinVazirani, DynamicCircuit) {
    dd::QubitCount nq = 127;
    auto           qc = std::make_unique<qc::BernsteinVazirani>(nq, true);
    qc->printStatistics(std::cout);

    // simulate the circuit
    auto        dd           = std::make_unique<dd::Package<>>(qc->getNqubits());
    std::size_t shots        = 1024;
    auto        measurements = simulate(qc.get(), dd->makeZeroState(qc->getNqubits()), dd, shots);

    for (const auto& [state, count]: measurements) {
        std::cout << state << ": " << count << std::endl;
    }

    // expect to obtain the hidden bitstring with certainty
    EXPECT_EQ(measurements[qc->expected], shots);
}

TEST_P(BernsteinVazirani, DynamicEquivalenceSimulation) {
    // get hidden bitstring
    auto s = qc::BitString(GetParam());

    // create standard BV circuit
    auto bv = std::make_unique<qc::BernsteinVazirani>(s);

    auto dd = std::make_unique<dd::Package<>>(bv->getNqubits());

    // remove final measurements to obtain statevector
    qc::CircuitOptimizer::removeFinalMeasurements(*bv);

    // simulate circuit
    auto e = simulate(bv.get(), dd->makeZeroState(bv->getNqubits()), dd);

    // create dynamic BV circuit
    auto dbv = std::make_unique<qc::BernsteinVazirani>(s, true);

    // transform dynamic circuits by first eliminating reset operations and afterwards deferring measurements
    qc::CircuitOptimizer::eliminateResets(*dbv);

    qc::CircuitOptimizer::deferMeasurements(*dbv);

    // remove final measurements to obtain statevector
    qc::CircuitOptimizer::removeFinalMeasurements(*dbv);

    // simulate circuit
    auto f = simulate(dbv.get(), dd->makeZeroState(dbv->getNqubits()), dd);

    // calculate fidelity between both results
    auto fidelity = dd->fidelity(e, f);
    std::cout << "Fidelity of both circuits: " << fidelity << std::endl;

    EXPECT_NEAR(fidelity, 1.0, 1e-4);
}
