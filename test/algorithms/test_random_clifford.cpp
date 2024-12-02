/*
 * Copyright (c) 2024 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/RandomCliffordCircuit.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <sstream>

class RandomClifford : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    RandomClifford, RandomClifford, testing::Range<std::size_t>(1U, 9U),
    [](const testing::TestParamInfo<RandomClifford::ParamType>& inf) {
      // Generate names for test cases
      const auto nqubits = inf.param;
      std::stringstream ss{};
      ss << static_cast<std::size_t>(nqubits) << "_qubits";
      return ss.str();
    });

TEST_P(RandomClifford, simulate) {
  const auto nq = GetParam();

  auto dd = std::make_unique<dd::Package<>>(nq);
  auto qc = qc::RandomCliffordCircuit(nq, nq * nq, 12345);
  auto in = dd->makeZeroState(nq);
  ASSERT_NO_THROW({ dd::simulate(&qc, in, *dd); });
  qc.printStatistics(std::cout);
}

TEST_P(RandomClifford, buildFunctionality) {
  const auto nq = GetParam();

  auto dd = std::make_unique<dd::Package<>>(nq);
  auto qc = qc::RandomCliffordCircuit(nq, nq * nq, 12345);
  ASSERT_NO_THROW({ dd::buildFunctionality(&qc, *dd); });
  qc.printStatistics(std::cout);
}
