#pragma once

#include "dd/Package.hpp"
#include "dd/Package_fwd.hpp"

#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace qc {
class QuantumComputation;
}

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

std::unique_ptr<SimulationExperiment>
benchmarkSimulate(const qc::QuantumComputation& qc);

std::unique_ptr<FunctionalityConstructionExperiment>
benchmarkFunctionalityConstruction(const qc::QuantumComputation& qc,
                                   bool recursive = false);

std::map<std::string, std::size_t>
benchmarkSimulateWithShots(const qc::QuantumComputation& qc, std::size_t shots);
} // namespace dd
