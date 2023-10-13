#include "algorithms/QFT.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"

#include "gtest/gtest.h"
#include <cmath>
#include <iostream>

class QFT : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {
    dd->decRef(sim);
    dd->decRef(func);
    dd->garbageCollect(true);
  }

  void SetUp() override {
    nqubits = GetParam();
    dd = std::make_unique<dd::Package<>>(nqubits);
  }

  std::size_t nqubits = 0;
  std::unique_ptr<dd::Package<>> dd;
  std::unique_ptr<qc::QFT> qc;
  qc::VectorDD sim{};
  qc::MatrixDD func{};
};

/// Findings from the QFT Benchmarks:
/// The DDpackage has to be able to represent all 2^n different amplitudes in
/// order to produce correct results The smallest entry seems to be closely
/// related to '1-cos(pi/2^(n-1))' The following CN::TOLERANCE values suffice up
/// until a certain number of qubits: 	10e-10	..	18 qubits
///		10e-11	..	20 qubits
///		10e-12	..	22 qubits
///		10e-13	..	23 qubits
/// The accuracy of double floating points allows for a minimal CN::TOLERANCE
/// value of 10e-15
///	Utilizing more qubits requires the use of fp=long double
constexpr std::size_t QFT_MAX_QUBITS = 20U;

INSTANTIATE_TEST_SUITE_P(QFT, QFT,
                         testing::Range<std::size_t>(0U, QFT_MAX_QUBITS + 1U,
                                                     3U),
                         [](const testing::TestParamInfo<QFT::ParamType>& inf) {
                           const auto nqubits = inf.param;
                           std::stringstream ss{};
                           ss << nqubits;
                           if (nqubits == 1) {
                             ss << "_qubit";
                           } else {
                             ss << "_qubits";
                           }
                           return ss.str();
                         });

TEST_P(QFT, Functionality) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::QFT>(nqubits, false); });
  // there should be no error building the functionality
  ASSERT_NO_THROW({ func = buildFunctionality(qc.get(), dd); });

  qc->printStatistics(std::cout);
  // QFT DD should consist of 2^n nodes
  ASSERT_EQ(func.size(), std::pow(2, nqubits));

  // Force garbage collection of compute table and complex table
  dd->garbageCollect(true);

  // top edge weight should equal sqrt(0.5)^n
  EXPECT_NEAR(func.w.real(),
              static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
              dd::EPS);

  // first row and first column should consist only of (1/sqrt(2))**nqubits
  for (std::uint64_t i = 0; i < std::pow(static_cast<long double>(2), nqubits);
       ++i) {
    auto c = func.getValueByIndex(0, i);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::EPS);
    EXPECT_NEAR(c.imag(), 0, dd::EPS);
    c = func.getValueByIndex(i, 0);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::EPS);
    EXPECT_NEAR(c.imag(), 0, dd::EPS);
  }
}

TEST_P(QFT, FunctionalityRecursive) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::QFT>(nqubits, false); });

  // there should be no error building the functionality
  ASSERT_NO_THROW({ func = buildFunctionalityRecursive(qc.get(), dd); });

  qc->printStatistics(std::cout);
  // QFT DD should consist of 2^n nodes
  ASSERT_EQ(func.size(), std::pow(2, nqubits));

  // Force garbage collection of compute table and complex table
  dd->garbageCollect(true);

  // top edge weight should equal sqrt(0.5)^n
  EXPECT_NEAR(func.w.real(),
              static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
              dd::EPS);

  // first row and first column should consist only of (1/sqrt(2))**nqubits
  for (std::uint64_t i = 0; i < std::pow(static_cast<long double>(2), nqubits);
       ++i) {
    auto c = func.getValueByIndex(0, i);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::EPS);
    EXPECT_NEAR(c.imag(), 0, dd::EPS);
    c = func.getValueByIndex(i, 0);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::EPS);
    EXPECT_NEAR(c.imag(), 0, dd::EPS);
  }
}

TEST_P(QFT, Simulation) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::QFT>(nqubits, false); });

  // there should be no error simulating the circuit
  ASSERT_NO_THROW({
    auto in = dd->makeZeroState(nqubits);
    sim = simulate(qc.get(), in, dd);
  });
  qc->printStatistics(std::cout);

  // QFT DD |0...0> sim should consist of n nodes
  ASSERT_EQ(sim.size(), nqubits + 1);

  // Force garbage collection of compute table and complex table
  dd->garbageCollect(true);

  // top edge weight should equal 1
  EXPECT_NEAR(sim.w.real(), 1, dd::EPS);
  EXPECT_NEAR(sim.w.imag(), 0, dd::EPS);

  // first column should consist only of sqrt(0.5)^n's
  for (std::uint64_t i = 0; i < std::pow(static_cast<long double>(2), nqubits);
       ++i) {
    auto c = sim.getValueByIndex(i);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::EPS);
    EXPECT_NEAR(c.imag(), 0, dd::EPS);
  }
}

TEST_P(QFT, FunctionalityRecursiveEquality) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::QFT>(nqubits, false); });

  // there should be no error building the functionality recursively
  ASSERT_NO_THROW({ func = buildFunctionalityRecursive(qc.get(), dd); });

  // there should be no error building the functionality regularly
  qc::MatrixDD funcRec{};
  ASSERT_NO_THROW({ funcRec = buildFunctionality(qc.get(), dd); });

  ASSERT_EQ(func, funcRec);
  dd->decRef(funcRec);
}

TEST_P(QFT, DynamicSimulation) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::QFT>(nqubits, true, true); });

  qc->printStatistics(std::cout);

  // simulate the circuit
  std::size_t shots = 8192U;
  auto measurements =
      simulate(qc.get(), dd->makeZeroState(qc->getNqubits()), dd, shots);

  for (const auto& [state, count] : measurements) {
    std::cout << state << ": " << count << "\n";
  }
  const std::size_t unique = measurements.size();

  nqubits = GetParam();
  const auto maxUnique = 1ULL << nqubits;
  if (maxUnique < shots) {
    shots = maxUnique;
  }
  const auto ratio = static_cast<double>(unique) / static_cast<double>(shots);
  std::cout << "Unique entries " << unique << " out of " << shots
            << " for a ratio of: " << ratio << "\n";

  // the number of unique entries should be close to the number of shots
  EXPECT_GE(ratio, 0.7);
}
