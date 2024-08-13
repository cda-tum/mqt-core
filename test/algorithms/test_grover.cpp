#include "algorithms/Grover.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>
#include <utility>

class Grover
    : public testing::TestWithParam<std::tuple<std::size_t, std::size_t>> {
protected:
  void TearDown() override {
    dd->decRef(sim);
    dd->decRef(func);
    dd->garbageCollect(true);
    // number of complex table entries after clean-up should equal 1
    EXPECT_EQ(dd->cn.realCount(), 1);
  }

  void SetUp() override {
    std::tie(nqubits, seed) = GetParam();
    dd = std::make_unique<dd::Package<>>(nqubits + 1);
  }

  std::size_t nqubits = 0;
  std::size_t seed = 0;
  std::unique_ptr<dd::Package<>> dd;
  std::unique_ptr<qc::Grover> qc;
  qc::VectorDD sim{};
  qc::MatrixDD func{};
};

constexpr std::size_t GROVER_MAX_QUBITS = 15;
constexpr std::size_t GROVER_NUM_SEEDS = 5;
constexpr dd::fp GROVER_ACCURACY = 1e-2;
constexpr dd::fp GROVER_GOAL_PROBABILITY = 0.9;

INSTANTIATE_TEST_SUITE_P(
    Grover, Grover,
    testing::Combine(
        testing::Range(static_cast<std::size_t>(2), GROVER_MAX_QUBITS + 1, 3),
        testing::Range(static_cast<std::size_t>(0), GROVER_NUM_SEEDS)),
    [](const testing::TestParamInfo<Grover::ParamType>& inf) {
      const auto nqubits = std::get<0>(inf.param);
      const auto seed = std::get<1>(inf.param);
      std::stringstream ss{};
      ss << nqubits + 1;
      if (nqubits == 0) {
        ss << "_qubit_";
      } else {
        ss << "_qubits_";
      }
      ss << seed;
      return ss.str();
    });

template <class Config>
qc::MatrixDD buildFunctionality(const qc::Grover* qc, dd::Package<Config>& dd) {
  qc::QuantumComputation groverIteration(qc->getNqubits());
  qc->oracle(groverIteration);
  qc->diffusion(groverIteration);

  auto iteration = buildFunctionality(&groverIteration, dd);

  auto e = iteration;
  dd.incRef(e);

  for (std::size_t i = 0U; i < qc->iterations - 1U; ++i) {
    auto f = dd.multiply(iteration, e);
    dd.incRef(f);
    dd.decRef(e);
    e = f;
    dd.garbageCollect();
  }

  qc::QuantumComputation setup(qc->getNqubits());
  qc->setup(setup);
  auto g = buildFunctionality(&setup, dd);
  auto f = dd.multiply(e, g);
  dd.incRef(f);
  dd.decRef(e);
  dd.decRef(g);
  e = f;

  dd.decRef(iteration);
  return e;
}

template <class Config>
qc::MatrixDD buildFunctionalityRecursive(const qc::Grover* qc,
                                         dd::Package<Config>& dd) {
  qc::QuantumComputation groverIteration(qc->getNqubits());
  qc->oracle(groverIteration);
  qc->diffusion(groverIteration);

  auto iter = buildFunctionalityRecursive(&groverIteration, dd);
  auto e = iter;
  std::bitset<128U> iterBits(qc->iterations);
  auto msb = static_cast<std::size_t>(std::floor(std::log2(qc->iterations)));
  auto f = iter;
  dd.incRef(f);
  bool zero = !iterBits[0U];
  for (std::size_t j = 1U; j <= msb; ++j) {
    auto tmp = dd.multiply(f, f);
    dd.incRef(tmp);
    dd.decRef(f);
    f = tmp;
    if (iterBits[j]) {
      if (zero) {
        dd.incRef(f);
        dd.decRef(e);
        e = f;
        zero = false;
      } else {
        auto g = dd.multiply(e, f);
        dd.incRef(g);
        dd.decRef(e);
        e = g;
        dd.garbageCollect();
      }
    }
  }
  dd.decRef(f);

  // apply state preparation setup
  qc::QuantumComputation statePrep(qc->getNqubits());
  qc->setup(statePrep);
  auto s = buildFunctionality(&statePrep, dd);
  auto tmp = dd.multiply(e, s);
  dd.incRef(tmp);
  dd.decRef(s);
  dd.decRef(e);
  e = tmp;

  return e;
}

TEST_P(Grover, Functionality) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::Grover>(nqubits, seed); });

  qc->printStatistics(std::cout);
  auto x = '1' + qc->expected;
  std::reverse(x.begin(), x.end());
  std::replace(x.begin(), x.end(), '1', '2');

  // there should be no error building the functionality
  ASSERT_NO_THROW({ func = buildFunctionality(qc.get(), *dd); });

  // amplitude of the searched-for entry should be 1
  auto c = func.getValueByPath(dd->qubits(), x);
  EXPECT_NEAR(std::abs(c.real()), 1, GROVER_ACCURACY);
  EXPECT_NEAR(std::abs(c.imag()), 0, GROVER_ACCURACY);
  const auto prob = std::norm(c);
  EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);
}

TEST_P(Grover, FunctionalityRecursive) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::Grover>(nqubits, seed); });

  qc->printStatistics(std::cout);
  auto x = '1' + qc->expected;
  std::reverse(x.begin(), x.end());
  std::replace(x.begin(), x.end(), '1', '2');

  // there should be no error building the functionality
  ASSERT_NO_THROW({ func = buildFunctionalityRecursive(qc.get(), *dd); });

  // amplitude of the searched-for entry should be 1
  auto c = func.getValueByPath(dd->qubits(), x);
  EXPECT_NEAR(std::abs(c.real()), 1, GROVER_ACCURACY);
  EXPECT_NEAR(std::abs(c.imag()), 0, GROVER_ACCURACY);
  const auto prob = std::norm(c);
  EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);
}

TEST_P(Grover, Simulation) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::Grover>(nqubits, seed); });

  qc->printStatistics(std::cout);
  auto in = dd->makeZeroState(nqubits + 1U);
  // there should be no error simulating the circuit
  const std::size_t shots = 1024;
  auto measurements = simulate(qc.get(), in, *dd, shots);

  for (const auto& [state, count] : measurements) {
    std::cout << state << ": " << count << "\n";
  }

  auto correctShots = measurements[qc->expected];
  auto probability =
      static_cast<double>(correctShots) / static_cast<double>(shots);

  EXPECT_GE(probability, GROVER_GOAL_PROBABILITY);
}
