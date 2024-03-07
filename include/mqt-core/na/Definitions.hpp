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
/**
 * @brief Class to store two-dimensional coordinates
 */
struct Point {
  std::int64_t x;
  std::int64_t y;
  Point(std::int64_t x, std::int64_t y) : x(x), y(y){};
  inline Point operator-(const Point& p) const {
    return {x - p.x, y - p.y};
  }
  inline Point operator-(const Point&& p) const {
    return {x - p.x, y - p.y};
  }
  inline Point operator+(const Point& p) const {
    return {x + p.x, y + p.y};
  }
  inline Point operator+(const Point&& p) const {
    return {x + p.x, y + p.y};
  }
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
  auto operator==(const Point& p) const -> bool {
    return p.x == x && p.y == y;
  }
};

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

static constexpr std::array<qc::OpType, 10> DIAGONAL_GATES = {
    qc::Barrier, qc::I,   qc::Z, qc::S,  qc::Sdg,
    qc::T,       qc::Tdg, qc::P, qc::RZ, qc::RZZ};

[[nodiscard]] inline auto isDiagonal(const qc::OpType& t) -> bool {
  return std::find(DIAGONAL_GATES.begin(), DIAGONAL_GATES.end(), t) !=
         DIAGONAL_GATES.end();
}

[[nodiscard]] inline auto isGlobal(const qc::Operation& op) -> bool {
  if (op.isStandardOperation()) {
    return op.getUsedQubits().size() == op.getNqubits();
  } else if (op.isCompoundOperation()) {
    const auto co = dynamic_cast<const qc::CompoundOperation&>(op);
    const auto& params = co.at(0)->getParameter();
    const auto& type = co.at(0)->getType();
    return op.getUsedQubits().size() == op.getNqubits() and std::all_of(co.cbegin(), co.cend(), [&](const auto& op) {
      return op->isStandardOperation() and op->getNcontrols() == 0 and op->getParameter() == params and op->getType() == type;
    });
  }
  return false;
}

[[nodiscard]] inline auto isSingleQubitGate(const qc::OpType& type) {
    switch (type) {
    case qc::OpType::U:
    case qc::OpType::U2:
    case qc::OpType::P:
    case qc::OpType::X:
    case qc::OpType::Y:
    case qc::OpType::Z:
    case qc::OpType::H:
    case qc::OpType::S:
    case qc::OpType::Sdg:
    case qc::OpType::T:
    case qc::OpType::SX:
    case qc::OpType::SXdg:
    case qc::OpType::Tdg:
    case qc::OpType::V:
    case qc::OpType::Vdg:
    case qc::OpType::RX:
    case qc::OpType::RY:
    case qc::OpType::RZ:
      return true;
    default:
      return false;
    }
  }

[[nodiscard]] inline auto isIndividual(const qc::Operation& op) -> bool {
  return op.getNcontrols() == 0 and isSingleQubitGate(op.getType());
}

} // namespace na

template <> struct std::hash<na::OpType> {
  std::size_t operator()(na::OpType const& t) const noexcept {
    std::size_t const h1 = std::hash<qc::OpType>{}(t.type);
    std::size_t const h2 = std::hash<std::size_t>{}(t.nctrl);
    return h1 ^ h2;
  }
};