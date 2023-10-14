#pragma once

#include "Definitions.hpp"
#include "operations/Control.hpp"

#include <map>

namespace qc {
class Permutation : public std::map<Qubit, Qubit> {
public:
  [[nodiscard]] inline Controls apply(const Controls& controls) const {
    if (empty()) {
      return controls;
    }
    Controls c{};
    for (const auto& control : controls) {
      c.emplace(at(control.qubit), control.type);
    }
    return c;
  }
  [[nodiscard]] inline Targets apply(const Targets& targets) const {
    if (empty()) {
      return targets;
    }
    Targets t{};
    for (const auto& target : targets) {
      t.emplace_back(at(target));
    }
    return t;
  }
};
} // namespace qc
