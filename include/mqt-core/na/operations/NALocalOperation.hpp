#pragma once

#include "Definitions.hpp"
#include "na/NADefinitions.hpp"
#include "na/operations/NAOperation.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace na {
class NALocalOperation : public NAOperation {
protected:
  FullOpType type;
  std::vector<qc::fp> params;
  std::vector<std::shared_ptr<Point>> positions;

public:
  NALocalOperation(const FullOpType& opType,
                   const std::vector<qc::fp>& parameter,
                   const std::vector<std::shared_ptr<Point>>& pos)
      : type(opType), params(parameter), positions(pos) {
    if (!opType.isSingleQubitType()) {
      throw std::invalid_argument("Operation is not single qubit.");
    }
    if (opType.isControlledType()) {
      throw std::logic_error("Control qubits are not supported.");
    }
  }
  explicit NALocalOperation(const FullOpType& opType,
                            const std::vector<std::shared_ptr<Point>>& pos)
      : NALocalOperation(opType, {}, pos) {}
  explicit NALocalOperation(const FullOpType& opType,
                            const std::vector<qc::fp>& parameters,
                            std::shared_ptr<Point> pos)
      : NALocalOperation(opType, parameters,
                         std::vector<std::shared_ptr<Point>>{std::move(pos)}) {}
  explicit NALocalOperation(const FullOpType& opType,
                            std::shared_ptr<Point> pos)
      : NALocalOperation(opType, {}, std::move(pos)) {}
  [[nodiscard]] auto
  getPositions() const -> const std::vector<std::shared_ptr<Point>>& {
    return positions;
  }
  [[nodiscard]] auto getParams() const -> const std::vector<qc::fp>& {
    return params;
  }
  [[nodiscard]] auto getType() const -> FullOpType { return type; }
  [[nodiscard]] auto isLocalOperation() const -> bool override { return true; }
  [[nodiscard]] auto toString() const -> std::string override;
  [[nodiscard]] auto clone() const -> std::unique_ptr<NAOperation> override {
    return std::make_unique<NALocalOperation>(*this);
  }
};
} // namespace na
