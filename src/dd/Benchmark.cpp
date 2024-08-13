#include "dd/Benchmark.hpp"

#include "algorithms/Grover.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "dd/statistics/PackageStatistics.hpp"
#include "ir/QuantumComputation.hpp"

#include <bitset>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <string>

namespace dd {

template <class Config>
MatrixDD buildFunctionality(const qc::Grover* qc, Package<Config>& dd) {
  QuantumComputation groverIteration(qc->getNqubits());
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

  QuantumComputation setup(qc->getNqubits());
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
MatrixDD buildFunctionalityRecursive(const qc::Grover* qc,
                                     Package<Config>& dd) {
  QuantumComputation groverIteration(qc->getNqubits());
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
