#include "eval/BuildTask.hpp"

#include "QuantumComputation.hpp"

BuildTask::BuildTask(std::unique_ptr<qc::QuantumComputation> qc)
    : qc(std::move(qc)) {}

std::string BuildTask::getIdentifier() const {
  return "build_" + qc->getName();
}
