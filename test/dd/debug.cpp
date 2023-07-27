#include "gtest/gtest.h"

#include "dd/Package.hpp"

using namespace qc;

TEST(DebugTest, debug) {
  const auto nqubits = 1U;
  auto matrix = dd::GateMatrix {
      dd::complex_one,
      dd::complex_zero,
      dd::complex_zero,
      dd::complex_zero
  };
  auto dd = std::make_unique<dd::Package<>>(nqubits);

  auto dd1 = dd->makeGateDD(matrix, nqubits, 0U);
  matrix[0] = dd::complex_zero;
  matrix[3] = dd::complex_one;
  auto dd2 = dd->makeGateDD(matrix, nqubits, 0U);
  const auto repetitions = 10U;
  for (auto i=0U; i<repetitions; ++i) {
    dd->multiply(dd1, dd2);
  }
  dd->garbageCollect(true);
  EXPECT_EQ(dd->mMemoryManager.getUsedCount(), 0U);
}