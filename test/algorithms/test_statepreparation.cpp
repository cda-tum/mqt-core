/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Definitions.hpp"
#include "algorithms/StatePreparation.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "ir/QuantumComputation.hpp"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

class StatePreparation
    : public testing::TestWithParam<std::vector<std::complex<double>>> {
protected:
  std::vector<std::complex<double>> amplitudes{};

  void TearDown() override {}
  void SetUp() override { amplitudes = GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(
    StatePreparation, StatePreparation,
    testing::Values(std::vector{std::complex{1 / std::sqrt(2)},
                                std::complex{-1 / std::sqrt(2)}},
                    std::vector<std::complex<double>>{
                        0, std::complex{1 / std::sqrt(2)},
                        std::complex{-1 / std::sqrt(2)}, 0}));

TEST_P(StatePreparation, StatePreparationCircuitSimulation) {
  const auto expectedAmplitudes = GetParam();
  qc::QuantumComputation qc;
  ASSERT_NO_THROW({ qc = qc::createStatePreparationCircuit(amplitudes); });
  auto dd = std::make_unique<dd::Package<>>(qc.getNqubits());
  qc::VectorDD e{};
  ASSERT_NO_THROW(
      { e = dd::simulate(qc, dd->makeZeroState(qc.getNqubits()), *dd); });
  auto result = e.getVector();
  ASSERT_EQ(expectedAmplitudes, result);
}

TEST_P(StatePreparation, StatePreparationCircuit) {}
