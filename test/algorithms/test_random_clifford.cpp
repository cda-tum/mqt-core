#include "algorithms/RandomCliffordCircuit.hpp"
#include "dd/Benchmark.hpp"

#include "gtest/gtest.h"
#include <string>

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

  auto qc = qc::RandomCliffordCircuit(nq, nq * nq, 12345);
  std::cout << qc << "\n";
  ASSERT_NO_THROW({
    auto out = dd::benchmarkSimulate(qc);
    EXPECT_TRUE(out->success());
  });
  qc.printStatistics(std::cout);
}

TEST_P(RandomClifford, buildFunctionality) {
  const auto nq = GetParam();

  auto qc = qc::RandomCliffordCircuit(nq, nq * nq, 12345);
  std::cout << qc << "\n";
  ASSERT_NO_THROW({
    auto out = dd::benchmarkFunctionalityConstruction(qc);
    EXPECT_TRUE(out->success());
  });
  qc.printStatistics(std::cout);
}
