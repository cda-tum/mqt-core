#include "CircuitOptimizer.hpp"
#include "algorithms/BernsteinVazirani.hpp"
#include "algorithms/Entanglement.hpp"
#include "algorithms/Grover.hpp"
#include "algorithms/QFT.hpp"
#include "algorithms/QPE.hpp"
#include "algorithms/WState.hpp"
#include "algorithms/RandomCliffordCircuit.hpp"


#include "dd/Benchmark.hpp"


#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

static constexpr bool ON_FEATURE_BRANCH = false;

namespace constants {
const int GLOBAL_SEED = 15;
}

// a function that parses a nlohmann::json from a file "results.json", populates
// it with the results of the current run and writes it back to the file
void verifyAndSaveSim(const std::string& name, const std::string& type,
                  qc::QuantumComputation& qc, const SimWithDDStats& simWithDdStats) {
EXPECT_NE(simWithDdStats.sim.p, nullptr);

nlohmann::json j;

  std::fstream file("results.json",
                    std::ios::in | std::ios::out | std::ios::ate);
  if (!file.is_open()) {
    std::ofstream outputFile("results.json");
    outputFile << nlohmann::json();
  } else if (file.tellg() == 0) {
    file << nlohmann::json();
  }
  file.close();

  std::ifstream ifs("results.json");
  ifs >> j;
  ifs.close();

  auto& entry = j[name][type][std::to_string(qc.getNqubits())]
                 [ON_FEATURE_BRANCH ? "feature" : "main"];

  entry["gate_count"] = qc.getNindividualOps();
  entry["runtime"] = simWithDdStats.runtime.count();

  // collect statistics from DD package
  entry["dd"] = simWithDdStats.ddStats;

  std::ofstream ofs("results.json");
  ofs << j.dump(2);
  ofs.close();
}

void verifyAndSaveFunc(const std::string& name, const std::string& type,
                      qc::QuantumComputation& qc, const FuncWithDDStats& funcWithDdStats) {
  EXPECT_NE(funcWithDdStats.func.p, nullptr);

  nlohmann::json j;

  std::fstream file("results.json",
                    std::ios::in | std::ios::out | std::ios::ate);
  if (!file.is_open()) {
    std::ofstream outputFile("results.json");
    outputFile << nlohmann::json();
  } else if (file.tellg() == 0) {
    file << nlohmann::json();
  }
  file.close();

  std::ifstream ifs("results.json");
  ifs >> j;
  ifs.close();

  auto& entry = j[name][type][std::to_string(qc.getNqubits())]
                 [ON_FEATURE_BRANCH ? "feature" : "main"];

  entry["gate_count"] = qc.getNindividualOps();
  entry["runtime"] = funcWithDdStats.runtime.count();

  // collect statistics from DD package
  entry["dd"] = funcWithDdStats.ddStats;

  std::ofstream ofs("results.json");
  ofs << j.dump(2);
  ofs.close();
}

class GHZEval : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {
    nqubits = GetParam();
    qc = std::make_unique<qc::Entanglement>(nqubits);
  }

  std::size_t nqubits = 0;
  std::unique_ptr<qc::Entanglement> qc;
};

INSTANTIATE_TEST_SUITE_P(GHZ, GHZEval,
                         testing::Values(256U, 512U, 1024U, 2048U, 4096U));

TEST_P(GHZEval, GHZSimulation) {
  const auto out = benchmarkSimulate(*qc);
  verifyAndSaveSim("GHZ", "Simulation", *qc, out);
}

TEST_P(GHZEval, GHZFunctionality) {
  const auto out = benchmarkBuildFunctionality(*qc);
  verifyAndSaveFunc("GHZ", "Functionality", *qc, out);
}

class WStateEval : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {
    nqubits = GetParam();
    qc = std::make_unique<qc::WState>(nqubits);
  }

  std::size_t nqubits = 0;
  std::unique_ptr<qc::WState> qc;
};

INSTANTIATE_TEST_SUITE_P(WState, WStateEval,
                         testing::Values(256U, 512U, 1024U, 2048U, 4096U));

TEST_P(WStateEval, WStateSimulation) {
  const auto out = benchmarkSimulate(*qc);
  verifyAndSaveSim("WState", "Simulation", *qc, out);
}

TEST_P(WStateEval, WStateFunctionality) {
  const auto out = benchmarkBuildFunctionality(*qc);
  verifyAndSaveFunc("WState", "Functionality", *qc, out);
}

// dynamic?
class BVEval : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {
    nqubits = GetParam();
    qc = std::make_unique<qc::BernsteinVazirani>(nqubits);
    qc::CircuitOptimizer::removeFinalMeasurements(*qc);
  }

  std::size_t nqubits = 0;
  std::unique_ptr<qc::BernsteinVazirani> qc;
};

INSTANTIATE_TEST_SUITE_P(BV, BVEval,
                         testing::Values(255U, 511U, 1023U, 2047U, 4095U));

TEST_P(BVEval, BVSimulation) {
  const auto out = benchmarkSimulate(*qc);
  verifyAndSaveSim("BV", "Simulation", *qc, out);
}

TEST_P(BVEval, BVFunctionality) {
  const auto out = benchmarkBuildFunctionality(*qc);
  verifyAndSaveFunc("BV", "Functionality", *qc, out);
}

class QFTEval : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {
    nqubits = GetParam();
    qc = std::make_unique<qc::QFT>(nqubits, false);
  }

  std::size_t nqubits = 0;
  std::unique_ptr<qc::QFT> qc;
};

INSTANTIATE_TEST_SUITE_P(QFT, QFTEval,
                         testing::Values(256U, 512U, 1024U, 2048U, 4096U));

TEST_P(QFTEval, QFTSimulation) {
  const auto out = benchmarkSimulate(*qc);
  verifyAndSaveSim("QFT", "Simulation", *qc, out);
}

class QFTEvalFunctionality : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {
    nqubits = GetParam();
    qc = std::make_unique<qc::QFT>(nqubits, false);
  }

  std::size_t nqubits = 0;
  std::unique_ptr<qc::QFT> qc;
};

INSTANTIATE_TEST_SUITE_P(QFT, QFTEvalFunctionality,
                         testing::Values(18U, 19U, 20U, 21U, 22U));

TEST_P(QFTEvalFunctionality, QFTFunctionality) {
  const auto out = benchmarkBuildFunctionality(*qc);
  verifyAndSaveFunc("QFT", "Functionality", *qc, out);
}

class GroverEval : public testing::TestWithParam<qc::Qubit> {
protected:
  void TearDown() override {}
  void SetUp() override {
    nqubits = GetParam();
    qc = std::make_unique<qc::Grover>(nqubits, 12345U);
    dd = std::make_unique<dd::Package<>>(qc->getNqubits());
  }

  qc::Qubit nqubits = 0;
  std::unique_ptr<dd::Package<>> dd;
  std::unique_ptr<qc::Grover> qc;
};

INSTANTIATE_TEST_SUITE_P(Grover, GroverEval,
                         testing::Values(27U, 31U, 35U, 39U, 41U));

TEST_P(GroverEval, GroverSimulator) {
  const auto start = std::chrono::high_resolution_clock::now();

  // apply state preparation setup
  qc::QuantumComputation statePrep(qc->getNqubits());
  qc->setup(statePrep);
  auto s = buildFunctionality(&statePrep, dd);
  auto e = dd->multiply(s, dd->makeZeroState(qc->getNqubits()));
  dd->incRef(e);
  dd->decRef(s);

  qc::QuantumComputation groverIteration(qc->getNqubits());
  qc->oracle(groverIteration);
  qc->diffusion(groverIteration);

  auto iter = buildFunctionalityRecursive(&groverIteration, dd);
  std::bitset<128U> iterBits(qc->iterations);
  auto msb = static_cast<std::size_t>(std::floor(std::log2(qc->iterations)));
  auto f = iter;
  dd->incRef(f);
  for (std::size_t j = 0U; j <= msb; ++j) {
    if (iterBits[j]) {
      auto g = dd->multiply(f, e);
      dd->incRef(g);
      dd->decRef(e);
      e = g;
      dd->garbageCollect();
    }
    if (j < msb) {
      auto tmp = dd->multiply(f, f);
      dd->incRef(tmp);
      dd->decRef(f);
      f = tmp;
    }
  }
  dd->decRef(f);
  const auto end = std::chrono::high_resolution_clock::now();
  EXPECT_NE(e.p, nullptr);
  const auto runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

  verifyAndSaveSim("Grover", "Simulation", *qc, {e, dd::getStatistics(dd.get()), runtime});
}

TEST_P(GroverEval, GroverFunctionality) {
  const auto out = benchmarkBuildFunctionality(*qc, true);
  verifyAndSaveFunc("Grover", "Functionality", *qc, out);
}

class QPEEval : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {
    nqubits = GetParam();
    qc = std::make_unique<qc::QPE>(nqubits, false);
    qc::CircuitOptimizer::removeFinalMeasurements(*qc);
  }

  std::size_t nqubits = 0;
  std::unique_ptr<qc::QPE> qc;
};

INSTANTIATE_TEST_SUITE_P(QPE, QPEEval,
                         testing::Values(14U, 15U, 16U, 17U, 18U));

TEST_P(QPEEval, QPESimulation) {
  const auto out = benchmarkSimulate(*qc);
  verifyAndSaveSim("QPE", "Simulation", *qc, out);
}

class QPEEvalFunctionality : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {
    nqubits = GetParam();
    qc = std::make_unique<qc::QPE>(nqubits, false);
    qc::CircuitOptimizer::removeFinalMeasurements(*qc);
  }

  std::size_t nqubits = 0;
  std::unique_ptr<qc::QPE> qc;
};

INSTANTIATE_TEST_SUITE_P(QPE, QPEEvalFunctionality,
                         testing::Values(7U, 8U, 9U, 10U, 11U));

TEST_P(QPEEvalFunctionality, QPEFunctionality) {
  const auto out = benchmarkBuildFunctionality(*qc);
  verifyAndSaveFunc("QPE", "Functionality", *qc, out);
}

class RandomCliffordEval : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {
    nqubits = GetParam();
    qc = std::make_unique<qc::RandomCliffordCircuit>(nqubits, nqubits*nqubits, constants::GLOBAL_SEED);
  }

  std::size_t nqubits = 0;
  std::unique_ptr<qc::RandomCliffordCircuit> qc;
};

INSTANTIATE_TEST_SUITE_P(RandomCliffordCircuit, RandomCliffordEval,
                         testing::Values(14U, 15U, 16U, 17U, 18U));

TEST_P(RandomCliffordEval, RandomCliffordSimulation) {
  const auto out = benchmarkSimulate(*qc);
  verifyAndSaveSim("RandomClifford", "Simulation", *qc, out);
}

class RandomCliffordEvalFunctionality : public testing::TestWithParam<std::size_t> {
protected:
  void TearDown() override {}
  void SetUp() override {
    nqubits = GetParam();
    qc = std::make_unique<qc::RandomCliffordCircuit>(nqubits, nqubits*nqubits, constants::GLOBAL_SEED);
  }

  std::size_t nqubits = 0;
  std::unique_ptr<qc::RandomCliffordCircuit> qc;
};

INSTANTIATE_TEST_SUITE_P(RandomCliffordCircuit, RandomCliffordEvalFunctionality,
                         testing::Values(7U, 8U, 9U, 10U, 11U));

TEST_P(RandomCliffordEvalFunctionality, RandomCliffordFunctionality) {
  const auto out = benchmarkBuildFunctionality(*qc);
  verifyAndSaveFunc("RandomClifford", "Functionality", *qc, out);
}

TEST(JSON, JSONTranspose) {
  std::ifstream ifs("results.json");
  nlohmann::json j;
  ifs >> j;
  ifs.close();

  nlohmann::json k;

  for (const auto& [algorithm, resultsA] : j.items()) {
    for (const auto& [type, resultsT] : resultsA.items()) {
      for (const auto& [nqubits, resultsN] : resultsT.items()) {
        for (const auto& [branch, resultsB] : resultsN.items()) {
          const auto& runtime = resultsB["runtime"];
          k[algorithm][type][nqubits]["runtime"][branch] = runtime;

          const auto& gateCount = resultsB["gate_count"];
          k[algorithm][type][nqubits]["gate_count"][branch] = gateCount;

          const auto& dd = resultsB["dd"];
          const auto& activeMemoryMiB = dd["active_memory_mib"];
          k[algorithm][type][nqubits]["dd"]["active_memory_mib"][branch] =
              activeMemoryMiB;
          const auto& peakMemoryMiB = dd["peak_memory_mib"];
          k[algorithm][type][nqubits]["dd"]["peak_memory_mib"][branch] =
              peakMemoryMiB;
          for (const auto& stat : {"matrix", "vector", "density_matrix",
                                   "real_numbers", "compute_tables"}) {
            for (const auto& [key, value] : dd[stat].items()) {
              if (value == "unused") {
                k[algorithm][type][nqubits]["dd"][stat][key][branch] = value;
                continue;
              }

              for (const auto& [key2, value2] : value.items()) {
                if ((std::strcmp(stat, "matrix") != 0 ||
                     std::strcmp(stat, "vector") != 0 ||
                     std::strcmp(stat, "density_matrix") != 0) &&
                    key == "unique_table") {
                  for (const auto& [key3, value3] : value2.items()) {
                    k[algorithm][type][nqubits]["dd"][stat][key][key2][key3]
                     [branch] = value3;
                  }
                  continue;
                }
                k[algorithm][type][nqubits]["dd"][stat][key][key2][branch] =
                    value2;
              }
            }
          }
        }
      }
    }
  }

  std::ofstream ofs("results_transposed.json");
  ofs << k.dump(2);
  ofs.close();
}

TEST(JSON, JSONReduce) {
  std::ifstream ifs("results_transposed.json");
  nlohmann::json j;
  ifs >> j;
  ifs.close();

  for (const auto& [algorithm, resultsA] : j.items()) {
    for (const auto& [type, resultsT] : resultsA.items()) {
      for (const auto& [nqubits, resultsN] : resultsT.items()) {
        auto& dd = resultsN["dd"];
        dd.erase("density_matrix");

        auto& computeTables = dd["compute_tables"];
        computeTables.erase("density_matrix_add");
        computeTables.erase("density_density_mult");
        computeTables.erase("density_noise_operations");
        computeTables.erase("stochastic_noise_operations");
        computeTables.erase("matrix_kronecker");
        computeTables.erase("vector_kronecker");
        computeTables.erase("vector_inner_product");
        computeTables.erase("matrix_conjugate_transpose");

        if (type == "Functionality") {
          dd.erase("vector");
          computeTables.erase("vector_add");
          computeTables.erase("matrix_vector_mult");
        }
      }
    }
  }

  std::ofstream ofs("results_reduced.json");
  ofs << j.dump(2);
  ofs.close();
}
