#pragma once

#include "na/NADefinitions.hpp"
#include "na/operations/NAOperation.hpp"
#include "operations/OpType.hpp"

#include <cmath>
namespace na {
class NAGlobalOperation : public NAOperation {
protected:
  FullOpType type;
  std::vector<qc::fp> params;

public:
  explicit NAGlobalOperation(const FullOpType type,
                             const std::vector<qc::fp>& params)
      : type(type), params(params) {
    if (!type.isSingleQubitType()) {
      throw std::invalid_argument("Operation is not single qubit.");
    }
  }
  explicit NAGlobalOperation(const FullOpType type)
      : NAGlobalOperation(type, {}) {}
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
