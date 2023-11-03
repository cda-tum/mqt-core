#include "dd/Benchmark.hpp"

#include "QuantumComputation.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "dd/statistics/PackageStatistics.hpp"

namespace dd {

std::unique_ptr<SimulationExperiment> benchmarkSimulate(const QuantumComputation& qc) {
  std::unique_ptr<SimulationExperiment> exp = std::make_unique<SimulationExperiment>();
  const auto nq = qc.getNqubits();
  exp->dd = std::make_unique<Package<>>(nq);
  const auto start = std::chrono::high_resolution_clock::now();
  const auto in = exp->dd->makeZeroState(nq);
  exp->sim = simulate(&qc, in, exp->dd);
  const auto end = std::chrono::high_resolution_clock::now();
  exp->runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  exp->stats = dd::getStatistics(exp->dd.get());
  return exp;
}

std::unique_ptr<FunctionalityConstructionExperiment>
benchmarkFunctionalityConstruction(const QuantumComputation& qc,
                                   const bool recursive) {
  std::unique_ptr<FunctionalityConstructionExperiment> exp = std::make_unique<FunctionalityConstructionExperiment>();
  const auto nq = qc.getNqubits();
  exp->dd = std::make_unique<Package<>>(nq);
  const auto start = std::chrono::high_resolution_clock::now();
  if (recursive) {
    exp->func = buildFunctionalityRecursive(&qc, exp->dd);
  } else {
    exp->func = buildFunctionality(&qc, exp->dd);
  }
  const auto end = std::chrono::high_resolution_clock::now();
  exp->runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  exp->stats = dd::getStatistics(exp->dd.get());
  return exp;
}

} // namespace dd
