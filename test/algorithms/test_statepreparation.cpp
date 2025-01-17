#include "Definitions.hpp"
#include "algorithms/StatePreparation.hpp"
#include "dd/Package.hpp"
#include "ir/QuantumComputation.hpp"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

class StatePreparation
    : public testing::TestWithParam<std::vector<std::complex<double>>> {
protected:
  std::vector<std::complex<double>> amplitudes{};

  void TearDown() override {}
  void SetUp() override { amplitudes = GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(
    StatePreparation, StatePreparation,
    testing::Values(std::vector{std::complex{1 / std::sqrt(2)},
                                std::complex{-1 / std::sqrt(2)}},
                    std::vector<std::complex<double>>{
                        0, std::complex{1 / std::sqrt(2)},
                        std::complex{-1 / std::sqrt(2)}, 0}));

TEST_P(StatePreparation, StatePreparationCircuitSimulation) {
  ASSERT_NO_THROW({ auto qc = qc::createStatePreparationCircuit(amplitudes); });
}

TEST_P(StatePreparation, StatePreparationCircuit) {}
