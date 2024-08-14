#include "Definitions.hpp"
#include "algorithms/WState.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/Simulation.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class WState : public testing::TestWithParam<qc::Qubit> {};

std::vector<std::string> generateWStateStrings(const std::size_t length) {
  std::vector<std::string> result;
  result.reserve(length);
  for (std::size_t i = 0U; i < length; ++i) {
    auto binaryString = std::string(length, '0');
    binaryString[i] = '1';
    result.emplace_back(binaryString);
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(
    WState, WState, testing::Range<qc::Qubit>(1U, 128U, 7U),
    [](const testing::TestParamInfo<WState::ParamType>& inf) {
      // Generate names for test cases
      const auto nqubits = inf.param;
      std::stringstream ss{};
      ss << nqubits << "_qubits";
      return ss.str();
    });

TEST_P(WState, FunctionTest) {
  const auto nq = GetParam();

  auto qc = qc::WState(nq);
  auto dd = std::make_unique<dd::Package<>>(qc.getNqubits());
  const std::size_t shots = 4096U;
  const auto measurements =
      simulate(&qc, dd->makeZeroState(qc.getNqubits()), *dd, shots);
  for (const auto& result : generateWStateStrings(nq)) {
    EXPECT_TRUE(measurements.find(result) != measurements.end());
  }
}

TEST_P(WState, RoutineFunctionTest) {
  const auto nq = GetParam();

  auto qc = qc::WState(nq);
  auto dd = std::make_unique<dd::Package<>>(qc.getNqubits());
  const dd::VectorDD e = simulate(&qc, dd->makeZeroState(qc.getNqubits()), *dd);
  const auto f = dd->makeWState(nq);

  EXPECT_EQ(e, f);
}

TEST(WState, WStateEdgeCasesTest) {
  auto dd = std::make_unique<dd::Package<>>(101);
  const auto tolerance = dd::RealNumber::eps;
  dd::ComplexNumbers::setTolerance(0.1);

  ASSERT_THROW(dd->makeWState(101), std::runtime_error);
  EXPECT_EQ(dd->makeWState(0), dd->makeBasisState(0, {dd::BasisStates::zero}));
  EXPECT_EQ(dd->makeWState(0), dd->makeBasisState(0, {dd::BasisStates::one}));
  EXPECT_EQ(dd->makeWState(1), dd->makeBasisState(1, {dd::BasisStates::one}));
  ASSERT_THROW(dd->makeWState(127), std::runtime_error);
  dd::ComplexNumbers::setTolerance(tolerance);
}
