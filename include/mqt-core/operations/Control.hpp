#pragma once

#include "Definitions.hpp"

#include <set>
#include <sstream>

namespace qc {
struct Control {
  enum class Type : bool { Pos = true, Neg = false };

  Qubit qubit{};
  Type type = Type::Pos;

  [[nodiscard]] std::string toString() const {
    std::ostringstream oss{};
    oss << "Control(qubit=" << qubit << ", type_=\"";
    if (type == Type::Pos) {
      oss << "Pos";
    } else {
      oss << "Neg";
    }
    oss << "\")";
    return oss.str();
  }

  // Explicitly allow implicit conversion from `Qubit` to `Control`
  // NOLINTNEXTLINE(google-explicit-constructor)
  Control(const Qubit q = {}, const Type t = Type::Pos) : qubit(q), type(t) {}
};

inline bool operator<(const Control& lhs, const Control& rhs) {
  return lhs.qubit < rhs.qubit ||
         (lhs.qubit == rhs.qubit && lhs.type < rhs.type);
}

inline bool operator==(const Control& lhs, const Control& rhs) {
  return lhs.qubit == rhs.qubit && lhs.type == rhs.type;
}

inline bool operator!=(const Control& lhs, const Control& rhs) {
  return !(lhs == rhs);
}

// this allows a set of controls to be indexed by a `Qubit`
struct CompareControl {
  using is_transparent [[maybe_unused]] = void;

  inline bool operator()(const Control& lhs, const Control& rhs) const {
    return lhs < rhs;
  }

  inline bool operator()(Qubit lhs, const Control& rhs) const {
    return lhs < rhs.qubit;
  }

  inline bool operator()(const Control& lhs, Qubit rhs) const {
    return lhs.qubit < rhs;
  }
};
using Controls = std::set<Control, CompareControl>;

inline namespace literals {
// User-defined literals require unsigned long long int
// NOLINTNEXTLINE(google-runtime-int)
inline Control operator""_pc(unsigned long long int q) {
  return {static_cast<Qubit>(q)};
}
// User-defined literals require unsigned long long int
// NOLINTNEXTLINE(google-runtime-int)
inline Control operator""_nc(unsigned long long int q) {
  return {static_cast<Qubit>(q), Control::Type::Neg};
}
} // namespace literals
} // namespace qc
