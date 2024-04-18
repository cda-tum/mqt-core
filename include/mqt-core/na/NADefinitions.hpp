//
// This file is part of the MQT CORE library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-core for more
// information.
//

#pragma once

#include "Definitions.hpp"
#include "operations/CompoundOperation.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <utility>
namespace na {
/// Class to store two-dimensional coordinates
struct Point {
  std::int64_t x;
  std::int64_t y;
  Point(std::int64_t x, std::int64_t y) : x(x), y(y){};
  inline Point operator-(const Point& p) const { return {x - p.x, y - p.y}; }
  inline Point operator-(const Point&& p) const { return {x - p.x, y - p.y}; }
  inline Point operator+(const Point& p) const { return {x + p.x, y + p.y}; }
  inline Point operator+(const Point&& p) const { return {x + p.x, y + p.y}; }
  [[nodiscard]] auto length() const -> std::uint64_t {
    return static_cast<std::uint64_t>(std::round(std::sqrt(x * x + y * y)));
  }
  [[nodiscard]] auto toString() const -> std::string {
    std::stringstream ss;
    ss << "(" << x << ", " << y << ")";
    return ss.str();
  }
  friend auto operator<<(std::ostream& os, const Point& obj) -> std::ostream& {
    os << obj.toString(); // Using toString() method
    return os;
  }
  auto operator==(const Point& p) const -> bool { return p.x == x && p.y == y; }
};
/// More specific operation type including the number of control qubits
struct OpType {
  qc::OpType type;
  std::size_t nctrl;
  [[nodiscard]] auto toString() const -> std::string {
    std::stringstream ss;
    for (std::size_t i = 0; i < nctrl; ++i) {
      ss << "c";
    }
    ss << qc::toString(type);
    return ss.str();
  }
  friend auto operator<<(std::ostream& os, const OpType& obj) -> std::ostream& {
    os << obj.toString(); // Using toString() method
    return os;
  }
  auto operator==(const OpType& t) const -> bool {
    return type == t.type && nctrl == t.nctrl;
  }
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
    return op.getUsedQubits().size() == nQubits and
           std::all_of(ops.cbegin(), ops.cend(), [&](const auto& op) {
             return op->isStandardOperation() and op->getNcontrols() == 0 and
                    op->getParameter() == params and op->getType() == type;
           });
  }
  return false;
}

} // namespace na

/// Hash function for OpType, e.g., for use in unordered_map
template <> struct std::hash<na::OpType> {
  std::size_t operator()(na::OpType const& t) const noexcept {
    std::size_t const h1 = std::hash<qc::OpType>{}(t.type);
    std::size_t const h2 = std::hash<std::size_t>{}(t.nctrl);
    return qc::combineHash(h1, h2);
  }
};