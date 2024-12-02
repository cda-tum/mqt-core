/*
 * Copyright (c) 2024 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Definitions.hpp"
#include "algorithms/BernsteinVazirani.hpp"
#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <sstream>

class BernsteinVazirani : public testing::TestWithParam<std::uint64_t> {
protected:
  void TearDown() override {}
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    BernsteinVazirani, BernsteinVazirani,
    testing::Values(0ULL,                      // Zero-Value
                    3ULL, 63ULL, 170ULL,       // 0-bit < hInt <= 8-bit
                    819ULL, 4032ULL, 33153ULL, // 8-bit < hInt <= 16-bit
                    87381ULL, 16777215ULL,
                    1234567891011ULL // 16-bit < hInt <= 32-bit
                    ),
    [](const testing::TestParamInfo<BernsteinVazirani::ParamType>& inf) {
      // Generate names for test cases
      const auto s = inf.param;
      std::stringstream ss{};
      ss << "bv_" << s;
      return ss.str();
    });

TEST_P(BernsteinVazirani, FunctionTest) {
  // get hidden bitstring
  auto s = qc::BitString(GetParam());

  // construct Bernstein Vazirani circuit
  const auto qc = qc::BernsteinVazirani(s);
  qc.printStatistics(std::cout);

  // simulate the circuit
  constexpr std::size_t shots = 1024;
  const auto measurements = dd::sample(qc, shots);

  // expect to obtain the hidden bitstring with certainty
  EXPECT_EQ(measurements.at(qc.expected), shots);
}

TEST_P(BernsteinVazirani, FunctionTestDynamic) {
  // get hidden bitstring
  const auto s = qc::BitString(GetParam());

  // construct Bernstein Vazirani circuit
  const auto qc = qc::BernsteinVazirani(s, true);
  qc.printStatistics(std::cout);

  // simulate the circuit
  constexpr std::size_t shots = 1024;
  const auto measurements = dd::sample(qc, shots);

  // expect to obtain the hidden bitstring with certainty
  EXPECT_EQ(measurements.at(qc.expected), shots);
}

TEST_F(BernsteinVazirani, LargeCircuit) {
  constexpr std::size_t nq = 127;
  const auto qc = qc::BernsteinVazirani(nq);

  // simulate the circuit
  constexpr std::size_t shots = 1024;
  const auto measurements = dd::sample(qc, shots);

  // expect to obtain the hidden bitstring with certainty
  EXPECT_EQ(measurements.at(qc.expected), shots);
}

TEST_F(BernsteinVazirani, DynamicCircuit) {
  constexpr std::size_t nq = 127;
  const auto qc = qc::BernsteinVazirani(nq, true);

  // simulate the circuit
  constexpr std::size_t shots = 1024;
  const auto measurements = dd::sample(qc, shots);

  // expect to obtain the hidden bitstring with certainty
  EXPECT_EQ(measurements.at(qc.expected), shots);
}

TEST_P(BernsteinVazirani, DynamicEquivalenceSimulation) {
  // get hidden bitstring
  auto s = qc::BitString(GetParam());

  // create standard BV circuit
  auto bv = qc::BernsteinVazirani(s);

  auto dd = std::make_unique<dd::Package<>>(bv.getNqubits());

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(bv);

  // simulate circuit
  auto e = dd::simulate(&bv, dd->makeZeroState(bv.getNqubits()), *dd);

  // create dynamic BV circuit
  auto dbv = qc::BernsteinVazirani(s, true);

  // transform dynamic circuits by first eliminating reset operations and
  // afterward deferring measurements
  qc::CircuitOptimizer::eliminateResets(dbv);
  qc::CircuitOptimizer::deferMeasurements(dbv);
  qc::CircuitOptimizer::backpropagateOutputPermutation(dbv);

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(dbv);

  // simulate circuit
  auto f = dd::simulate(&dbv, dd->makeZeroState(dbv.getNqubits()), *dd);

  // calculate fidelity between both results
  auto fidelity = dd->fidelity(e, f);
  EXPECT_NEAR(fidelity, 1.0, 1e-4);
}
