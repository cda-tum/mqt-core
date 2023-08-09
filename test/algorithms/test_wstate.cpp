#include "algorithms/WState.hpp"
#include "dd/Simulation.hpp"

#include "gtest/gtest.h"
#include <iostream>
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
  const std::size_t shots = 1024;
  const auto measurements =
      simulate(&qc, dd->makeZeroState(qc.getNqubits()), dd, shots);
  for (const auto& result : generateWStateStrings(nq)) {
    EXPECT_TRUE(measurements.find(result) != measurements.end());
  }
}
