#pragma once

#include "Definitions.hpp"
#include "operations/CompoundOperation.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <sstream>
#include <string>

namespace na {
/// Class to store two-dimensional coordinates
struct Point {
  std::int64_t x;
  std::int64_t y;
  Point(const std::int64_t xp, const std::int64_t yp) : x(xp), y(yp) {};
  Point(const Point& p) = default;
  virtual ~Point() = default;
  Point& operator=(const Point& p) = default;
  Point operator-(const Point& p) const { return {x - p.x, y - p.y}; }
  Point operator-(const Point&& p) const { return {x - p.x, y - p.y}; }
  Point operator+(const Point& p) const { return {x + p.x, y + p.y}; }
  Point operator+(const Point&& p) const { return {x + p.x, y + p.y}; }
  [[nodiscard]] auto length() const -> std::uint64_t {
    return static_cast<std::uint64_t>(std::round(std::sqrt(x * x + y * y)));
  }
  [[nodiscard]] auto toString() const -> std::string {
    std::stringstream ss;
    ss << "(" << x << ", " << y << ")";
    return ss.str();
  }
  friend auto operator<<(std::ostream& os, const Point& obj) -> std::ostream& {
    return os << obj.toString(); // Using toString() method
  }
  [[nodiscard]] auto operator==(const Point& other) const -> bool {
    return x == other.x && y == other.y;
  }
  [[maybe_unused]] [[nodiscard]] auto
  getEuclideanDistance(const Point& c) const {
    const auto delta = *this - c;
    return delta.length();
  }

  [[maybe_unused]] [[nodiscard]] std::int64_t
  getManhattanDistanceX(const Point& c) const {
    if (x > c.x) {
      return x - c.x;
    }
    return c.x - x;
  }
  [[maybe_unused]] [[nodiscard]] std::int64_t
  getManhattanDistanceY(const Point& c) const {
    if (y > c.y) {
      return y - c.y;
    }
    return c.y - y;
  }
};
/// More specific operation type including the number of control qubits
struct FullOpType {
  qc::OpType type;
  std::size_t nControls;
  [[nodiscard]] auto toString() const -> std::string {
    return std::string(nControls, 'c') + qc::toString(type);
  }
  friend auto operator<<(std::ostream& os,
                         const FullOpType& obj) -> std::ostream& {
    return os << obj.toString(); // Using toString() method
  }
  [[nodiscard]] auto operator==(const FullOpType& other) const -> bool {
    return type == other.type && nControls == other.nControls;
  }
  [[nodiscard]] auto isSingleQubitType() const -> bool {
    return isSingleQubitGate(type);
  }
  [[nodiscard]] auto isTwoQubitType() const -> bool {
    return isTwoQubitGate(type);
  }
  [[nodiscard]] auto isControlledType() const -> bool { return nControls > 0; }
};

/**
 * @brief Checks whether a gate is global.
 * @details A StandardOperation is global if it acts on all qubits.
 * A CompoundOperation is global if all its sub-operations are
 * StandardOperations of the same type with the same parameters acting on all
 * qubits. The latter is what a QASM line like `ry(π) q;` is translated to in
 * MQT-core. All other operations are not global.
 */
[[nodiscard]] inline auto isGlobal(const qc::Operation& op,
                                   const std::size_t nQubits) -> bool {
  if (op.isStandardOperation()) {
    return op.getUsedQubits().size() == nQubits;
  }
  if (op.isCompoundOperation()) {
    const auto ops = dynamic_cast<const qc::CompoundOperation&>(op);
    const auto& params = ops.at(0)->getParameter();
    const auto& type = ops.at(0)->getType();
    return op.getUsedQubits().size() == nQubits &&
           std::all_of(ops.cbegin(), ops.cend(), [&](const auto& operation) {
             return operation->isStandardOperation() &&
                    operation->getNcontrols() == 0 &&
                    operation->getType() == type &&
                    operation->getParameter() == params;
           });
  }
  return false;
}

} // namespace na

/// Hash function for OpType, e.g., for use in unordered_map
template <> struct std::hash<na::FullOpType> {
  std::size_t operator()(na::FullOpType const& t) const noexcept {
    std::size_t const h1 = std::hash<qc::OpType>{}(t.type);
    std::size_t const h2 = std::hash<std::size_t>{}(t.nControls);
    return qc::combineHash(h1, h2);
  }
};

/// Hash function for Point, e.g., for use in unordered_map
template <> struct std::hash<na::Point> {
  std::size_t operator()(const na::Point& p) const noexcept {
    const std::size_t h1 = std::hash<decltype(p.x)>{}(p.x);
    const std::size_t h2 = std::hash<decltype(p.y)>{}(p.y);
    return qc::combineHash(h1, h2);
  }
};
