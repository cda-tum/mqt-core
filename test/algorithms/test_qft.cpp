#include "algorithms/QFT.hpp"
#include "dd/Benchmark.hpp"
#include "dd/FunctionalityConstruction.hpp"

#include "gtest/gtest.h"
#include <cmath>
#include <iostream>

class QFT : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}

  void SetUp() override { nqubits = GetParam(); }

  std::size_t nqubits = 0;
  std::unique_ptr<qc::QFT> qc;
  std::unique_ptr<dd::Experiment> exp;
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

static const size_t INITIAL_COMPLEX_COUNT = 1;

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

  exp = dd::benchmarkFunctionalityConstruction(*qc);
  auto* expFunc =
      dynamic_cast<dd::FunctionalityConstructionExperiment*>(exp.get());
  assert(expFunc != nullptr);
  auto func = expFunc->func;
  auto dd = std::move(exp->dd);

  qc->printStatistics(std::cout);
  // QFT DD should consist of 2^n nodes
  ASSERT_EQ(func.size(), std::pow(2, nqubits));

  // Force garbage collection of compute table and complex table
  dd->garbageCollect(true);

  // the final DD should store all 2^n different amplitudes
  // since only positive real values are stored in the complex table
  // this number has to be divided by 4
  ASSERT_EQ(dd->cn.realCount(),
            static_cast<std::size_t>(std::ceil(std::pow(2, nqubits) / 4)));

  // top edge weight should equal sqrt(0.5)^n
  EXPECT_NEAR(dd::RealNumber::val(func.w.r),
              static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
              dd::RealNumber::eps);

  // first row and first column should consist only of (1/sqrt(2))**nqubits
  for (std::uint64_t i = 0; i < std::pow(static_cast<long double>(2), nqubits);
       ++i) {
    auto c = func.getValueByIndex(0, i);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::RealNumber::eps);
    EXPECT_NEAR(c.imag(), 0, dd::RealNumber::eps);
    c = func.getValueByIndex(i, 0);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::RealNumber::eps);
    EXPECT_NEAR(c.imag(), 0, dd::RealNumber::eps);
  }
  dd->decRef(func);
  dd->garbageCollect(true);
  // number of complex table entries after clean-up should equal initial
  // number of entries
  EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
}

TEST_P(QFT, FunctionalityRecursive) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::QFT>(nqubits, false); });

  exp = dd::benchmarkFunctionalityConstruction(*qc, true);
  auto* expFunc =
      dynamic_cast<dd::FunctionalityConstructionExperiment*>(exp.get());
  assert(expFunc != nullptr);
  auto func = expFunc->func;
  auto dd = std::move(exp->dd);

  qc->printStatistics(std::cout);
  // QFT DD should consist of 2^n nodes
  ASSERT_EQ(func.size(), std::pow(2, nqubits));

  // Force garbage collection of compute table and complex table
  dd->garbageCollect(true);

  // the final DD should store all 2^n different amplitudes
  // since only positive real values are stored in the complex table
  // this number has to be divided by 4
  ASSERT_EQ(dd->cn.realCount(),
            static_cast<std::size_t>(std::ceil(std::pow(2, nqubits) / 4)));

  // top edge weight should equal sqrt(0.5)^n
  EXPECT_NEAR(dd::RealNumber::val(func.w.r),
              static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
              dd::RealNumber::eps);

  // first row and first column should consist only of (1/sqrt(2))**nqubits
  for (std::uint64_t i = 0; i < std::pow(static_cast<long double>(2), nqubits);
       ++i) {
    auto c = func.getValueByIndex(0, i);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::RealNumber::eps);
    EXPECT_NEAR(c.imag(), 0, dd::RealNumber::eps);
    c = func.getValueByIndex(i, 0);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::RealNumber::eps);
    EXPECT_NEAR(c.imag(), 0, dd::RealNumber::eps);
  }
  dd->decRef(func);
  dd->garbageCollect(true);
  // number of complex table entries after clean-up should equal initial
  // number of entries
  EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
}

TEST_P(QFT, Simulation) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::QFT>(nqubits, false); });

  exp = dd::benchmarkSimulate(*qc);
  auto* expSim = dynamic_cast<dd::SimulationExperiment*>(exp.get());
  auto sim = expSim->sim;
  auto dd = std::move(exp->dd);

  qc->printStatistics(std::cout);

  // QFT DD |0...0> sim should consist of n nodes
  ASSERT_EQ(sim.size(), nqubits + 1);

  // Force garbage collection of compute table and complex table
  dd->garbageCollect(true);

  // top edge weight should equal 1
  EXPECT_NEAR(dd::RealNumber::val(sim.w.r), 1, dd::RealNumber::eps);
  EXPECT_NEAR(dd::RealNumber::val(sim.w.i), 0, dd::RealNumber::eps);

  // first column should consist only of sqrt(0.5)^n's
  for (std::uint64_t i = 0; i < std::pow(static_cast<long double>(2), nqubits);
       ++i) {
    auto c = sim.getValueByIndex(i);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::RealNumber::eps);
    EXPECT_NEAR(c.imag(), 0, dd::RealNumber::eps);
  }
  dd->decRef(sim);
  dd->garbageCollect(true);
  // number of complex table entries after clean-up should equal initial
  // number of entries
  EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
}

TEST_P(QFT, FunctionalityRecursiveEquality) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::QFT>(nqubits, false); });

  // there should be no error building the functionality recursively
  exp = dd::benchmarkFunctionalityConstruction(*qc);
  auto* expFunc =
      dynamic_cast<dd::FunctionalityConstructionExperiment*>(exp.get());
  assert(expFunc != nullptr);
  auto func = expFunc->func;
  auto dd = std::move(exp->dd);

  // there should be no error building the functionality regularly
  auto funcRec = buildFunctionalityRecursive(qc.get(), *dd);

  ASSERT_EQ(func, funcRec);
  dd->decRef(funcRec);
  dd->decRef(func);
  dd->garbageCollect(true);
  // number of complex table entries after clean-up should equal initial
  // number of entries
  EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
}

TEST_P(QFT, DynamicSimulation) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::QFT>(nqubits, true, true); });
  auto dd = std::make_unique<dd::Package<>>(nqubits);
  qc->printStatistics(std::cout);

  // simulate the circuit
  std::size_t shots = 8192U;
  auto measurements = dd::benchmarkSimulateWithShots(*qc, shots);
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
  dd->garbageCollect(true);
  // number of complex table entries after clean-up should equal initial
  // number of entries
  EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
}
