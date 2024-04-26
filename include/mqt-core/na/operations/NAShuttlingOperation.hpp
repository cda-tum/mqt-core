#pragma once

#include "na/NADefinitions.hpp"
#include "na/operations/NAOperation.hpp"
#include "operations/OpType.hpp"

#include <cmath>
#include <utility>
#include <vector>
namespace na {

enum ShuttleType : std::uint8_t { LOAD, MOVE, STORE };

class NAShuttlingOperation : public NAOperation {
protected:
  ShuttleType type;
  std::vector<std::shared_ptr<Point>> start;
  std::vector<std::shared_ptr<Point>> end;

public:
  explicit NAShuttlingOperation(
      const ShuttleType type, const std::vector<std::shared_ptr<Point>>& start,
      const std::vector<std::shared_ptr<Point>>& end)
      : type(type), start(start), end(end) {
    if (start.size() != end.size()) {
      throw std::logic_error("Shuttling operation must have the same number of "
                             "start and end qubits.");
    }
  }
  explicit NAShuttlingOperation(const ShuttleType type,
                                std::shared_ptr<Point> start,
                                std::shared_ptr<Point> end)
      : NAShuttlingOperation(
            type, std::vector<std::shared_ptr<Point>>{std::move(start)},
            std::vector<std::shared_ptr<Point>>{std::move(end)}) {}
  [[nodiscard]] auto getType() const -> ShuttleType { return type; }
  [[nodiscard]] auto
  getStart() const -> const std::vector<std::shared_ptr<Point>>& {
    return start;
  }
  [[nodiscard]] auto
  getEnd() const -> const std::vector<std::shared_ptr<Point>>& {
    return end;
  }
  [[nodiscard]] auto isShuttlingOperation() const -> bool override {
    return true;
  }
  [[nodiscard]] auto toString() const -> std::string override;
  [[nodiscard]] auto clone() const -> std::unique_ptr<NAOperation> override {
    return std::make_unique<NAShuttlingOperation>(*this);
  }
};
} // namespace na
