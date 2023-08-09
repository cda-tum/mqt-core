#include "algorithms/WState.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"

#include "gtest/gtest.h"

#include <iostream>
#include <vector>

class WState : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {}
};

std::vector<std::string> generateWStateStrings(std::size_t length) {
  std::vector<std::string> result;
  for (int i = 0; i < (1 << length); i++) {
    int countOnes = 0;
    std::string binaryString;
    for (size_t j = 0; j < length; j++) {
      if (((i >> j) & 1) != 0) {
        countOnes++;
        binaryString += "1";
      } else {
        binaryString += "0";
      }
    }
    if (countOnes == 1) {
      result.push_back(binaryString);
    }
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(
    WState, WState, testing::Range<std::size_t>(2U, 30U, 6U),
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
  auto measurements =
      simulate(&qc, dd->makeZeroState(qc.getNqubits()), dd, shots);
  auto results = generateWStateStrings(nq);

  for (const auto& result : results) {
    EXPECT_TRUE(measurements.find(result) != measurements.end());
  }
}
