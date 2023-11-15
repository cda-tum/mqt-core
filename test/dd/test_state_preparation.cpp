#include "dd/Package.hpp"

#include <gtest/gtest.h>

class StatePreparation : public testing::TestWithParam<qc::Qubit> {};

extern std::vector<std::string> generateWStateStrings(std::size_t length);

INSTANTIATE_TEST_SUITE_P(
    StatePreparation, StatePreparation, testing::Range<qc::Qubit>(1U, 128U, 7U),
    [](const testing::TestParamInfo<StatePreparation::ParamType>& inf) {
      // Generate names for test cases
      const auto nqubits = inf.param;
      std::stringstream ss{};
      ss << nqubits << "_qubits";
      return ss.str();
    });

TEST_P(StatePreparation, GHZStateTest) {
  const auto nq = GetParam();

  auto dd = std::make_unique<dd::Package<>>(nq);
  const auto ghz = dd->makeGHZState(nq);
  EXPECT_EQ(ghz.getValueByIndex(0), dd::SQRT2_2);
  EXPECT_EQ(ghz.getValueByPath(std::string(nq, '1')), dd::SQRT2_2);
}

TEST_P(StatePreparation, WStateTest) {
  const auto nq = GetParam();

  auto dd = std::make_unique<dd::Package<>>(nq);
  const auto w = dd->makeWState(nq);
  for (const auto& wStateString : generateWStateStrings(nq)) {
    EXPECT_NEAR(w.getValueByPath(wStateString).real(), 1. / std::sqrt(nq),
                dd::RealNumber::eps);
    EXPECT_EQ(w.getValueByPath(wStateString).imag(), 0.);
  }
}

TEST(DDPackageTest, GHZStateEdgeCasesTest) {
  auto dd = std::make_unique<dd::Package<>>(3);

  EXPECT_EQ(dd->makeGHZState(0),
            dd->makeBasisState(0, {dd::BasisStates::zero}));
  EXPECT_EQ(dd->makeGHZState(0), dd->makeBasisState(0, {dd::BasisStates::one}));
  ASSERT_THROW(dd->makeGHZState(6), std::runtime_error);
}

TEST(DDPackageTest, WStateEdgeCasesTest) {
  auto dd = std::make_unique<dd::Package<>>(3);

  EXPECT_EQ(dd->makeWState(0), dd->makeBasisState(0, {dd::BasisStates::zero}));
  EXPECT_EQ(dd->makeWState(0), dd->makeBasisState(0, {dd::BasisStates::one}));
  EXPECT_EQ(dd->makeWState(1), dd->makeBasisState(1, {dd::BasisStates::one}));
  ASSERT_THROW(dd->makeWState(6), std::runtime_error);
}
