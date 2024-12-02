/*
 * Copyright (c) 2024 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/Entanglement.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

class Entanglement : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {
    nq = GetParam();
    dd = std::make_unique<dd::Package<>>(nq);
  }
  std::size_t nq{};
  std::unique_ptr<dd::Package<>> dd;
};

INSTANTIATE_TEST_SUITE_P(
    Entanglement, Entanglement, testing::Range<std::size_t>(2U, 90U, 7U),
    [](const testing::TestParamInfo<Entanglement::ParamType>& inf) {
      // Generate names for test cases
      const auto nqubits = inf.param;
      std::stringstream ss{};
      ss << nqubits << "_qubits";
      return ss.str();
    });

TEST_P(Entanglement, FunctionTest) {
  const auto qc = qc::Entanglement(nq);
  const auto e = dd::buildFunctionality(&qc, *dd);
  ASSERT_EQ(qc.getNops(), nq);
  const auto r = dd->multiply(e, dd->makeZeroState(nq));
  ASSERT_EQ(r.getValueByPath(nq, std::string(nq, '0')), dd::SQRT2_2);
  ASSERT_EQ(r.getValueByPath(nq, std::string(nq, '1')), dd::SQRT2_2);
}

TEST_P(Entanglement, GHZRoutineFunctionTest) {
  const auto qc = qc::Entanglement(nq);
  const auto e = dd::simulate(&qc, dd->makeZeroState(nq), *dd);
  const auto f = dd->makeGHZState(nq);
  EXPECT_EQ(e, f);
}

TEST(Entanglement, GHZStateEdgeCasesTest) {
  auto dd = std::make_unique<dd::Package<>>(3);

  EXPECT_EQ(dd->makeGHZState(0),
            dd->makeBasisState(0, {dd::BasisStates::zero}));
  EXPECT_EQ(dd->makeGHZState(0), dd->makeBasisState(0, {dd::BasisStates::one}));
  ASSERT_THROW(dd->makeGHZState(6), std::runtime_error);
}
