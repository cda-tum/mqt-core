#include "CircuitOptimizer.hpp"
#include "algorithms/BernsteinVazirani.hpp"
#include "dd/Benchmark.hpp"
#include "dd/Simulation.hpp"

#include "gtest/gtest.h"

class BernsteinVazirani : public testing::TestWithParam<std::uint64_t> {
protected:
  void TearDown() override {}
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    BernsteinVazirani, BernsteinVazirani,
    testing::Values(0ULL,                      // Zero-Value
                    3ULL, 63ULL, 170ULL,       // 0-bit < hInt <= 8-bit
                    819ULL, 4032ULL, 33153ULL, // 8-bit < hInt <= 16-bit
                    87381ULL, 16777215ULL,
                    1234567891011ULL // 16-bit < hInt <= 32-bit
                    ),
    [](const testing::TestParamInfo<BernsteinVazirani::ParamType>& inf) {
      // Generate names for test cases
      const auto s = inf.param;
      std::stringstream ss{};
      ss << "bv_" << s;
      return ss.str();
    });

TEST_P(BernsteinVazirani, FunctionTest) {
  // get hidden bitstring
  auto s = qc::BitString(GetParam());

  // construct Bernstein Vazirani circuit
  auto qc = qc::BernsteinVazirani(s);
  qc.printStatistics(std::cout);

  // simulate the circuit
  auto dd = std::make_unique<dd::Package<>>(qc.getNqubits());
  const std::size_t shots = 1024;
  auto measurements =
      simulate(&qc, dd->makeZeroState(qc.getNqubits()), dd, shots);

  for (const auto& [state, count] : measurements) {
    std::cout << state << ": " << count << "\n";
  }

  // expect to obtain the hidden bitstring with certainty
  EXPECT_EQ(measurements[qc.expected], shots);
}

TEST_P(BernsteinVazirani, FunctionTestDynamic) {
  // get hidden bitstring
  auto s = qc::BitString(GetParam());

  // construct Bernstein Vazirani circuit
  auto qc = qc::BernsteinVazirani(s, true);
  qc.printStatistics(std::cout);

  // simulate the circuit
  auto dd = std::make_unique<dd::Package<>>(qc.getNqubits());
  const std::size_t shots = 1024;
  auto measurements =
      simulate(&qc, dd->makeZeroState(qc.getNqubits()), dd, shots);

  for (const auto& [state, count] : measurements) {
    std::cout << state << ": " << count << "\n";
  }

  // expect to obtain the hidden bitstring with certainty
  EXPECT_EQ(measurements[qc.expected], shots);
}

TEST_F(BernsteinVazirani, LargeCircuit) {
  const std::size_t nq = 127;
  auto qc = qc::BernsteinVazirani(nq);
  qc.printStatistics(std::cout);

  // simulate the circuit
  auto dd = std::make_unique<dd::Package<>>(qc.getNqubits());
  const std::size_t shots = 1024;
  auto measurements =
      simulate(&qc, dd->makeZeroState(qc.getNqubits()), dd, shots);

  for (const auto& [state, count] : measurements) {
    std::cout << state << ": " << count << "\n";
  }

  // expect to obtain the hidden bitstring with certainty
  EXPECT_EQ(measurements[qc.expected], shots);
}

TEST_F(BernsteinVazirani, DynamicCircuit) {
  const std::size_t nq = 127;
  auto qc = qc::BernsteinVazirani(nq, true);
  qc.printStatistics(std::cout);

  // simulate the circuit
  auto dd = std::make_unique<dd::Package<>>(qc.getNqubits());
  const std::size_t shots = 1024;
  auto measurements =
      simulate(&qc, dd->makeZeroState(qc.getNqubits()), dd, shots);

  for (const auto& [state, count] : measurements) {
    std::cout << state << ": " << count << "\n";
  }

  // expect to obtain the hidden bitstring with certainty
  EXPECT_EQ(measurements[qc.expected], shots);
}

TEST_P(BernsteinVazirani, DynamicEquivalenceSimulation) {
  // get hidden bitstring
  auto s = qc::BitString(GetParam());

  // create standard BV circuit
  auto bv = qc::BernsteinVazirani(s);

  auto dd = std::make_unique<dd::Package<>>(bv.getNqubits());

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(bv);

  // simulate circuit
  auto e = simulate(&bv, dd->makeZeroState(bv.getNqubits()), dd);

  // create dynamic BV circuit
  auto dbv = qc::BernsteinVazirani(s, true);

  // transform dynamic circuits by first eliminating reset operations and
  // afterwards deferring measurements
  qc::CircuitOptimizer::eliminateResets(dbv);
  qc::CircuitOptimizer::deferMeasurements(dbv);

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(dbv);

  // simulate circuit
  auto f = simulate(&dbv, dd->makeZeroState(dbv.getNqubits()), dd);

  // calculate fidelity between both results
  auto fidelity = dd->fidelity(e, f);
  std::cout << "Fidelity of both circuits: " << fidelity << "\n";

  EXPECT_NEAR(fidelity, 1.0, 1e-4);
}

TEST_F(BernsteinVazirani, BVBenchmarkSimulate) {
  auto nqubits = 20;
  auto qc = std::make_unique<qc::BernsteinVazirani>(nqubits);
  qc::CircuitOptimizer::removeFinalMeasurements(*qc);
  const auto out = dd::benchmarkSimulate(*qc);
  EXPECT_NE(out->sim.p, nullptr);
}

TEST_F(BernsteinVazirani, BVBenchmarkFunctionality) {
  auto nqubits = 20;
  auto qc = std::make_unique<qc::BernsteinVazirani>(nqubits);
  qc::CircuitOptimizer::removeFinalMeasurements(*qc);
  const auto out = dd::benchmarkFunctionalityConstruction(*qc);
  EXPECT_NE(out->func.p, nullptr);
}
