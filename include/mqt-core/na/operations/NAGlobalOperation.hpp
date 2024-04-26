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
    if (!isSingleQubitGate(type.type)) {
      throw std::invalid_argument("Operation is not single qubit.");
    }
  }
  explicit NAGlobalOperation(const FullOpType type) : NAGlobalOperation(type, {}) {}
  [[nodiscard]] auto getParams() const -> const std::vector<qc::fp>& {
    return params;
  }
  [[nodiscard]] auto isGlobalOperation() const -> bool override { return true; }
  [[nodiscard]] auto toString() const -> std::string override {
    std::stringstream ss;
    ss << type;
    if (!params.empty()) {
      ss << "(";
      for (const auto& p : params) {
        ss << p << ", ";
      }
      ss.seekp(-2, std::ios_base::end);
      ss << ")";
    }
    ss << ";\n";
    return ss.str();
  }
  [[nodiscard]] auto clone() const -> std::unique_ptr<NAOperation> override {
    return std::make_unique<NAGlobalOperation>(*this);
  }
};
} // namespace na
