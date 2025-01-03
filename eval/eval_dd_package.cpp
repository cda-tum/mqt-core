/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/BernsteinVazirani.hpp"
#include "algorithms/Entanglement.hpp"
#include "algorithms/Grover.hpp"
#include "algorithms/QFT.hpp"
#include "algorithms/QPE.hpp"
#include "algorithms/RandomCliffordCircuit.hpp"
#include "algorithms/WState.hpp"
#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "dd/statistics/PackageStatistics.hpp"
#include "ir/QuantumComputation.hpp"

#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace dd {

struct Experiment {
  std::unique_ptr<Package<>> dd;
  nlohmann::basic_json<> stats = nlohmann::basic_json<>::object();
  std::chrono::duration<double> runtime{};

  Experiment() = default;
  Experiment(const Experiment&) = delete;
  Experiment(Experiment&&) = default;
  Experiment& operator=(const Experiment&) = delete;
  Experiment& operator=(Experiment&&) = default;
  virtual ~Experiment() = default;

  [[nodiscard]] virtual bool success() const noexcept { return false; }
};

struct SimulationExperiment final : public Experiment {
  SimulationExperiment() = default;
  qc::VectorDD sim{};

  [[nodiscard]] bool success() const noexcept override {
    return sim.p != nullptr;
  }
};

struct FunctionalityConstructionExperiment final : public Experiment {
  qc::MatrixDD func{};

  [[nodiscard]] bool success() const noexcept override {
    return func.p != nullptr;
  }
};

namespace {
template <class Config>
MatrixDD buildFunctionality(const qc::Grover* qc, Package<Config>& dd) {
  QuantumComputation groverIteration(qc->getNqubits());
  qc->oracle(groverIteration);
  qc->diffusion(groverIteration);

  auto iteration = buildFunctionality(&groverIteration, dd);

  auto e = iteration;
  dd.incRef(e);
  for (std::size_t i = 0U; i < qc->iterations - 1U; ++i) {
    e = dd.applyOperation(iteration, e);
  }
  dd.decRef(iteration);

  QuantumComputation setup(qc->getNqubits());
  qc->setup(setup);
  const auto g = buildFunctionality(&setup, dd);
  e = dd.applyOperation(e, g);
  dd.decRef(g);
  return e;
}

template <class Config>
MatrixDD buildFunctionalityRecursive(const qc::Grover* qc,
                                     Package<Config>& dd) {
  QuantumComputation groverIteration(qc->getNqubits());
  qc->oracle(groverIteration);
  qc->diffusion(groverIteration);

  auto iter = buildFunctionalityRecursive(&groverIteration, dd);
  auto e = iter;
  std::bitset<128U> iterBits(qc->iterations);
  const auto msb =
      static_cast<std::size_t>(std::floor(std::log2(qc->iterations)));
  auto f = iter;
  dd.incRef(f);
  bool zero = !iterBits[0U];
  for (std::size_t j = 1U; j <= msb; ++j) {
    f = dd.applyOperation(f, f);
    if (iterBits[j]) {
      if (zero) {
        dd.incRef(f);
        dd.decRef(e);
        e = f;
        zero = false;
      } else {
        e = dd.applyOperation(e, f);
      }
    }
  }
  dd.decRef(f);

  // apply state preparation setup
  qc::QuantumComputation statePrep(qc->getNqubits());
  qc->setup(statePrep);
  const auto s = buildFunctionality(&statePrep, dd);
  return dd.applyOperation(e, s);
}

std::unique_ptr<SimulationExperiment>
benchmarkSimulate(const QuantumComputation& qc) {
  auto exp = std::make_unique<SimulationExperiment>();
  const auto nq = qc.getNqubits();
  exp->dd = std::make_unique<Package<>>(nq);
  const auto start = std::chrono::high_resolution_clock::now();
  const auto in = exp->dd->makeZeroState(nq);
  exp->sim = simulate(&qc, in, *(exp->dd));
  const auto end = std::chrono::high_resolution_clock::now();
  exp->runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  exp->stats = dd::getStatistics(exp->dd.get());
  return exp;
}

std::unique_ptr<FunctionalityConstructionExperiment>
benchmarkFunctionalityConstruction(const QuantumComputation& qc,
                                   const bool recursive = false) {
  auto exp = std::make_unique<FunctionalityConstructionExperiment>();
  const auto nq = qc.getNqubits();
  exp->dd = std::make_unique<Package<>>(nq);
  const auto start = std::chrono::high_resolution_clock::now();

  if (const auto* grover = dynamic_cast<const Grover*>(&qc)) {
    if (recursive) {
      exp->func = buildFunctionalityRecursive(grover, *(exp->dd));
    } else {
      exp->func = buildFunctionality(grover, *(exp->dd));
    }
  } else {
    if (recursive) {
      exp->func = buildFunctionalityRecursive(&qc, *(exp->dd));
    } else {
      exp->func = buildFunctionality(&qc, *(exp->dd));
    }
  }
  const auto end = std::chrono::high_resolution_clock::now();
  exp->runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  exp->stats = dd::getStatistics(exp->dd.get());
  return exp;
}
} // namespace

static constexpr std::size_t SEED = 42U;

class BenchmarkDDPackage {
protected:
  void verifyAndSave(const std::string& name, const std::string& type,
                     qc::QuantumComputation& qc, const Experiment& exp) const {

    const std::string& filename = "results_" + inputFilename + ".json";

    nlohmann::basic_json<> j;
    std::fstream file(filename, std::ios::in | std::ios::out | std::ios::ate);
    if (!file.is_open()) {
      std::ofstream outputFile(filename);
      outputFile << nlohmann::json();
    } else if (file.tellg() == 0) {
      file << nlohmann::json();
    }
    file.close();

    std::ifstream ifs(filename);
    ifs >> j;
    ifs.close();

    auto& entry = j[name][type][std::to_string(qc.getNqubits())];

    entry["runtime"] = exp.runtime.count();

    // collect statistics from DD package
    entry["dd"] = exp.stats;

    std::ofstream ofs(filename);
    ofs << j.dump(2U);
    ofs.close();
  }

  std::string inputFilename;

  void runGHZ() const {
    constexpr std::array nqubits = {256U, 512U, 1024U, 2048U, 4096U};
    std::cout << "Running GHZ Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::Entanglement(nq);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("GHZ", "Simulation", qc, *exp);
    }
    std::cout << "Running GHZ Functionality..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::Entanglement(nq);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("GHZ", "Functionality", qc, *exp);
    }
  }

  void runWState() const {
    constexpr std::array nqubits = {256U, 512U, 1024U, 2048U, 4096U};
    std::cout << "Running WState Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::WState(nq);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("WState", "Simulation", qc, *exp);
    }
    std::cout << "Running WState Functionality..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::WState(nq);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("WState", "Functionality", qc, *exp);
    }
  }

  void runBV() const {
    constexpr std::array nqubits = {255U, 511U, 1023U, 2047U, 4095U};
    std::cout << "Running BV Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::BernsteinVazirani(nq);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("BV", "Simulation", qc, *exp);
    }
    std::cout << "Running BV Functionality..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::BernsteinVazirani(nq);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("BV", "Functionality", qc, *exp);
    }
  }

  void runQFT() const {
    constexpr std::array nqubitsSim = {256U, 512U, 1024U, 2048U, 4096U};
    std::cout << "Running QFT Simulation..." << '\n';
    for (const auto& nq : nqubitsSim) {
      auto qc = qc::QFT(nq, false);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("QFT", "Simulation", qc, *exp);
    }
    constexpr std::array nqubitsFunc = {18U, 19U, 20U, 21U, 22U};
    std::cout << "Running QFT Functionality..." << '\n';
    for (const auto& nq : nqubitsFunc) {
      auto qc = qc::QFT(nq, false);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("QFT", "Functionality", qc, *exp);
    }
  }

  void runGrover() {
    constexpr std::array nqubits = {27U, 31U, 35U, 39U, 41U};
    std::cout << "Running Grover Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = std::make_unique<qc::Grover>(nq, SEED);
      auto dd = std::make_unique<dd::Package<>>(qc->getNqubits());
      const auto start = std::chrono::high_resolution_clock::now();

      // apply state preparation setup
      qc::QuantumComputation statePrep(qc->getNqubits());
      qc->setup(statePrep);
      auto s = buildFunctionality(&statePrep, *dd);
      auto e = dd->applyOperation(s, dd->makeZeroState(qc->getNqubits()));

      qc::QuantumComputation groverIteration(qc->getNqubits());
      qc->oracle(groverIteration);
      qc->diffusion(groverIteration);

      auto iter = buildFunctionalityRecursive(&groverIteration, *dd);
      std::bitset<128U> iterBits(qc->iterations);
      const auto msb =
          static_cast<std::size_t>(std::floor(std::log2(qc->iterations)));
      auto f = iter;
      dd->incRef(f);
      for (std::size_t j = 0U; j <= msb; ++j) {
        if (iterBits[j]) {
          e = dd->applyOperation(f, e);
        }
        if (j < msb) {
          f = dd->applyOperation(f, f);
        }
      }
      dd->decRef(f);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto runtime =
          std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                    start);
      auto exp = std::make_unique<SimulationExperiment>();
      exp->dd = std::move(dd);
      exp->sim = e;
      exp->runtime = runtime;
      exp->stats = dd::getStatistics(exp->dd.get());

      verifyAndSave("Grover", "Simulation", *qc, *exp);
    }

    std::cout << "Running Grover Functionality..." << '\n';
    for (const auto& nq : nqubits) {

      auto qc = std::make_unique<qc::Grover>(nq, SEED);
      auto exp = benchmarkFunctionalityConstruction(*qc, true);
      verifyAndSave("Grover", "Functionality", *qc, *exp);
    }
  }

  void runQPE() const {
    constexpr std::array nqubitsSim = {14U, 15U, 16U, 17U, 18U};
    std::cout << "Running QPE Simulation..." << '\n';
    for (const auto& nq : nqubitsSim) {
      auto qc = qc::QPE(nq, false);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("QPE", "Simulation", qc, *exp);
    }
    std::cout << "Running QPE Functionality..." << '\n';
    constexpr std::array nqubitsFunc = {7U, 8U, 9U, 10U, 11U};
    for (const auto& nq : nqubitsFunc) {
      auto qc = qc::QPE(nq, false);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("QPE", "Functionality", qc, *exp);
    }
  }

  void runRandomClifford() const {
    constexpr std::array<std::size_t, 5> nqubitsSim = {14U, 15U, 16U, 17U, 18U};
    std::cout << "Running RandomClifford Simulation..." << '\n';
    for (const auto& nq : nqubitsSim) {
      auto qc = qc::RandomCliffordCircuit(nq, nq * nq, SEED);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("RandomClifford", "Simulation", qc, *exp);
    }
    std::cout << "Running RandomClifford Functionality..." << '\n';
    constexpr std::array<std::size_t, 5> nqubitsFunc = {7U, 8U, 9U, 10U, 11U};
    for (const auto& nq : nqubitsFunc) {
      auto qc = qc::RandomCliffordCircuit(nq, nq * nq, SEED);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("RandomClifford", "Functionality", qc, *exp);
    }
  }

public:
  explicit BenchmarkDDPackage(std::string filename)
      : inputFilename(std::move(filename)) {};

  void runAll() {
    runGHZ();
    runWState();
    runBV();
    runQFT();
    runGrover();
    runQPE();
    runRandomClifford();
  }
};

} // namespace dd

int main(const int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Exactly one argument is required to name the results file."
              << '\n';
    return 1;
  }
  try {
    auto run = dd::BenchmarkDDPackage(
        argv[1]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    run.runAll();
  } catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << '\n';
    return 1;
  }
  std::cout << "Benchmarks done." << '\n';
  return 0;
}
