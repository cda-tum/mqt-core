#include "algorithms/Entanglement.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"

#include "gtest/gtest.h"
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

  auto dd = std::make_unique<dd::Package<>>(nq);
  auto qc = qc::Entanglement(nq);
  auto e = buildFunctionality(&qc, dd);

  ASSERT_EQ(qc.getNops(), nq);
  const qc::VectorDD r = dd->multiply(e, dd->makeZeroState(nq));

  ASSERT_EQ(r.getValueByPath(std::string(nq, '0')), dd::SQRT2_2);
  ASSERT_EQ(r.getValueByPath(std::string(nq, '1')), dd::SQRT2_2);
}

TEST_P(Entanglement, GHZRoutineFunctionTest) {
  const auto nq = GetParam();

  auto qc = qc::Entanglement(nq);
  auto dd = std::make_unique<dd::Package<>>(nq);
  const dd::VectorDD e = simulate(&qc, dd->makeZeroState(qc.getNqubits()), dd);
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
