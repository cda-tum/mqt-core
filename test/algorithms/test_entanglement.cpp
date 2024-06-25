#include "algorithms/Entanglement.hpp"
#include "dd/Benchmark.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "dd/Package_fwd.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

class Entanglement : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {}
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
  const auto nq = GetParam();

  auto qc = qc::Entanglement(nq);
  auto result = dd::benchmarkFunctionalityConstruction(qc);
  auto e = result->func;

  ASSERT_EQ(qc.getNops(), nq);
  const qc::VectorDD r = result->dd->multiply(e, result->dd->makeZeroState(nq));

  ASSERT_EQ(r.getValueByPath(nq, std::string(nq, '0')), dd::SQRT2_2);
  ASSERT_EQ(r.getValueByPath(nq, std::string(nq, '1')), dd::SQRT2_2);
}

TEST_P(Entanglement, GHZRoutineFunctionTest) {
  const auto nq = GetParam();

  auto qc = qc::Entanglement(nq);
  auto exp = dd::benchmarkSimulate(qc);
  auto e = exp->sim;
  const auto f = exp->dd->makeGHZState(nq);

  EXPECT_EQ(e, f);
}

TEST(Entanglement, GHZStateEdgeCasesTest) {
  auto dd = std::make_unique<dd::Package<>>(3);

  EXPECT_EQ(dd->makeGHZState(0),
            dd->makeBasisState(0, {dd::BasisStates::zero}));
  EXPECT_EQ(dd->makeGHZState(0), dd->makeBasisState(0, {dd::BasisStates::one}));
  ASSERT_THROW(dd->makeGHZState(6), std::runtime_error);
}
