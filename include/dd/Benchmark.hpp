#pragma once

#include "dd/Package_fwd.hpp"

#include <chrono>
#include <nlohmann/json.hpp>

namespace qc {
class QuantumComputation;
}

namespace dd {

struct Experiment {
  std::unique_ptr<Package<>> dd{};
  nlohmann::json stats = nlohmann::json::object();
  std::chrono::duration<double> runtime{};

  virtual ~Experiment() = default;

  [[nodiscard]] virtual bool success() const noexcept { return false; }
};

struct SimulationExperiment : public Experiment {
  SimulationExperiment() = default;
  qc::VectorDD sim{};

  [[nodiscard]] bool success() const noexcept override {
    return sim.p != nullptr;
  }
};

struct FunctionalityConstructionExperiment : public Experiment {
  qc::MatrixDD func{};

  [[nodiscard]] bool success() const noexcept override {
    return func.p != nullptr;
  }
};

std::unique_ptr<SimulationExperiment> benchmarkSimulate(const qc::QuantumComputation& qc);

std::unique_ptr<FunctionalityConstructionExperiment>
benchmarkFunctionalityConstruction(const qc::QuantumComputation& qc,
                                   bool recursive = false);
} // namespace dd
