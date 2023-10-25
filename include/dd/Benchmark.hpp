#pragma once

#include "QuantumComputation.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"

namespace constants {
  const std::size_t SHOTS = 1024;
}

inline std::map<std::string, std::size_t> benchmarkSimulate(const qc::QuantumComputation& qc) {
        auto nq = qc.getNqubits();
        auto dd = std::make_unique<dd::Package<>>(nq + 1);
        auto in = dd->makeZeroState(nq + 1U);
        auto measurements = simulate(&qc, in, dd, constants::SHOTS);
        return measurements;
}

inline qc::MatrixDD benchmarkBuildFunctionality(const qc::QuantumComputation& qc) {
  auto nq = qc.getNqubits();
  auto dd = std::make_unique<dd::Package<>>(nq);
  auto func = buildFunctionality(&qc, dd);
  return func;
}
