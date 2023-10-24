#include "dd/Package.hpp"

#include <gtest/gtest.h>

TEST(DDPackageTest, GHZStateTest) {
  auto dd = std::make_unique<dd::Package<>>(3);
  auto ghz = dd->makeGHZState(3);

  // build state vector for (|000> + |111>) * 1/sqrt(2)
  const dd::CVec state = {dd::SQRT2_2, 0, 0, 0, 0, 0, 0, dd::SQRT2_2};
  const auto stateDD = dd->makeStateFromVector(state);

  EXPECT_EQ(ghz, stateDD);
}

TEST(DDPackage, WStateTest) {
  auto dd = std::make_unique<dd::Package<>>(3);
  auto wState = dd->makeWState(3);

  // build state vector for (|001> + |010> + |100>) * 1/sqrt(3)
  const dd::CVec state = {
      0, std::sqrt(3) / 3, std::sqrt(3) / 3, 0, std::sqrt(3) / 3, 0, 0, 0};
  const auto stateDD = dd->makeStateFromVector(state);

  EXPECT_EQ(wState, stateDD);
}

TEST(DDPackageTest, GHZStateEdgeCasesTest) {
  auto dd = std::make_unique<dd::Package<>>(3);

  ASSERT_THROW(dd->makeGHZState(0), std::runtime_error);
  ASSERT_THROW(dd->makeGHZState(6), std::runtime_error);
}

TEST(DDPackageTest, WStateEdgeCasesTest) {
  auto dd = std::make_unique<dd::Package<>>(3);

  EXPECT_EQ(dd->makeWState(1), dd->makeBasisState(1, {dd::BasisStates::one}));
  ASSERT_THROW(dd->makeWState(0), std::runtime_error);
  ASSERT_THROW(dd->makeWState(6), std::runtime_error);
}
