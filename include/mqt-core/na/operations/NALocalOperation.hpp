#pragma once

#include "na/NADefinitions.hpp"
#include "na/operations/NAOperation.hpp"
#include "operations/OpType.hpp"

#include <cmath>
#include <utility>
namespace na {
class NALocalOperation : public NAOperation {
protected:
  FullOpType type;
  std::vector<qc::fp> params;
  std::vector<std::shared_ptr<Point>> positions;

public:
  NALocalOperation(const FullOpType& type, const std::vector<qc::fp>& params,
                   const std::vector<std::shared_ptr<Point>>& positions)
      : type(type), params(params), positions(positions) {
    if (!type.isSingleQubitType()) {
      throw std::invalid_argument("Operation is not single qubit.");
    }
    if (type.isControlledType()) {
      throw std::logic_error("Control qubits are not supported.");
    }
  }
  explicit NALocalOperation(
      const FullOpType& type,
      const std::vector<std::shared_ptr<Point>>& positions)
      : NALocalOperation(type, {}, positions) {}
  explicit NALocalOperation(const FullOpType& type,
                            const std::vector<qc::fp>& params,
                            std::shared_ptr<Point> position)
      : NALocalOperation(
            type, params,
            std::vector<std::shared_ptr<Point>>{std::move(position)}) {}
  explicit NALocalOperation(const FullOpType& type,
                            std::shared_ptr<Point> position)
      : NALocalOperation(type, {}, std::move(position)) {}
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
