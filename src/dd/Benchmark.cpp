#include "dd/Benchmark.hpp"

#include "QuantumComputation.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "dd/statistics/PackageStatistics.hpp"

#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <string>

namespace dd {

std::unique_ptr<SimulationExperiment>
benchmarkSimulate(const QuantumComputation& qc) {
  std::unique_ptr<SimulationExperiment> exp =
      std::make_unique<SimulationExperiment>();
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
                                   const bool recursive) {
  std::unique_ptr<FunctionalityConstructionExperiment> exp =
      std::make_unique<FunctionalityConstructionExperiment>();
  const auto nq = qc.getNqubits();
  exp->dd = std::make_unique<Package<>>(nq);
  const auto start = std::chrono::high_resolution_clock::now();
  if (recursive) {
    exp->func = buildFunctionalityRecursive(&qc, *(exp->dd));
  } else {
    exp->func = buildFunctionality(&qc, *(exp->dd));
  }
  const auto end = std::chrono::high_resolution_clock::now();
  exp->runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  exp->stats = dd::getStatistics(exp->dd.get());
  return exp;
}

std::map<std::string, std::size_t>
benchmarkSimulateWithShots(const qc::QuantumComputation& qc,
                           const std::size_t shots) {
  auto nq = qc.getNqubits();
  auto dd = std::make_unique<dd::Package<>>(nq);
  auto in = dd->makeZeroState(nq);
  auto measurements = simulate(&qc, in, *dd, shots);
  return measurements;
}

} // namespace dd
