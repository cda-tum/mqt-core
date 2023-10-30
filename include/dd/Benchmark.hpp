#pragma once

#include "QuantumComputation.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"
#include "dd/Package.hpp"
#include "dd/statistics/PackageStatistics.hpp"

#include <chrono>

namespace constants {
const std::size_t SHOTS = 1024;
}

struct SimWithDDStats{
    dd::VectorDD sim{};
    nlohmann::json ddStats = nlohmann::json::object();
    std::chrono::duration<double> runtime{};
};

struct FuncWithDDStats{
    dd::MatrixDD func{};
    nlohmann::json ddStats = nlohmann::json::object();
    std::chrono::duration<double> runtime{};
};
//combine these two?

inline SimWithDDStats
benchmarkSimulate(const qc::QuantumComputation& qc) {
  auto nq = qc.getNqubits();
  auto dd = std::make_unique<dd::Package<>>(nq);
  auto in = dd->makeZeroState(nq);
  const auto start = std::chrono::high_resolution_clock::now();
  auto sim = simulate(&qc, in, dd);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  SimWithDDStats simWithDDStats;
  simWithDDStats.sim = sim;
  auto ddStats = dd::getStatistics(dd.get());
  simWithDDStats.ddStats = ddStats;
  simWithDDStats.runtime = runtime;
  return simWithDDStats;
}

[[maybe_unused]] inline std::map<std::string, std::size_t>
benchmarkSimulateWithShots(const qc::QuantumComputation& qc) {
  auto nq = qc.getNqubits();
  auto dd = std::make_unique<dd::Package<>>(nq + 1);
  auto in = dd->makeZeroState(nq + 1U);
  auto measurements = simulate(&qc, in, dd, constants::SHOTS);
  return measurements;
}

inline FuncWithDDStats
benchmarkBuildFunctionality(const qc::QuantumComputation& qc, bool recursive = false) {
  auto nq = qc.getNqubits();
  auto dd = std::make_unique<dd::Package<>>(nq);
  const auto start = std::chrono::high_resolution_clock::now();
  dd::MatrixDD func;
  if (recursive) {
    func = buildFunctionalityRecursive(&qc, dd);
  } else {
    func = buildFunctionality(&qc, dd);
  }
  const auto end = std::chrono::high_resolution_clock::now();
  const auto runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  FuncWithDDStats funcWithDDStats;
  funcWithDDStats.func = func;
  auto ddStats = dd::getStatistics(dd.get());
  funcWithDDStats.ddStats = ddStats;
  funcWithDDStats.runtime = runtime;
  return funcWithDDStats;
}
