#include "eval/SimulationTask.hpp"

#include "QuantumComputation.hpp"

SimulationTask::SimulationTask(std::unique_ptr<qc::QuantumComputation> qc)
    : qc(std::move(qc)) {}

std::string SimulationTask::getIdentifier() const {
  return "sim_" + qc->getName();
}
