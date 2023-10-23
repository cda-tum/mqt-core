#pragma once

#include "Task.hpp"

#include <memory>

namespace qc {
class QuantumComputation;
}

class BuildTask : public Task {
public:
  explicit BuildTask() = default;
  explicit BuildTask(std::unique_ptr<qc::QuantumComputation> qc);

  [[nodiscard]] std::string getIdentifier() const override;

  [[nodiscard]] const std::unique_ptr<qc::QuantumComputation>& getQc() const {
    return qc;
  };

protected:
  std::unique_ptr<qc::QuantumComputation> qc;
};
