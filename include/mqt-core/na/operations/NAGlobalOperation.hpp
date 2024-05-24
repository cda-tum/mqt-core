#pragma once

#include "Definitions.hpp"
#include "na/NADefinitions.hpp"
#include "na/operations/NAOperation.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace na {
class NAGlobalOperation : public NAOperation {
protected:
  FullOpType type;
  std::vector<qc::fp> params;

public:
  explicit NAGlobalOperation(const FullOpType opType,
                             const std::vector<qc::fp>& parameters)
      : type(opType), params(parameters) {
    if (!opType.isSingleQubitType()) {
      throw std::invalid_argument("Operation is not single qubit.");
    }
  }
  explicit NAGlobalOperation(const FullOpType opType)
      : NAGlobalOperation(opType, {}) {}
  [[nodiscard]] auto getParams() const -> const std::vector<qc::fp>& {
    return params;
  }
  [[nodiscard]] auto isGlobalOperation() const -> bool override { return true; }
  [[nodiscard]] auto toString() const -> std::string override;
  [[nodiscard]] auto clone() const -> std::unique_ptr<NAOperation> override {
    return std::make_unique<NAGlobalOperation>(*this);
  }
};
} // namespace na
