/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "entities/Atom.hpp"
#include "entities/Location.hpp"
#include "operations/Op.hpp"

#include <iterator>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace na {
class NAComputation : std::vector<Op> {
protected:
  std::vector<Atom> atoms;
  std::unordered_map<const Atom*, Location> initialLocations;

public:
  NAComputation() = default;
  NAComputation(const NAComputation& qc) = default;
  NAComputation(NAComputation&& qc) noexcept = default;
  NAComputation& operator=(const NAComputation& qc) = default;
  NAComputation& operator=(NAComputation&& qc) noexcept = default;
  virtual ~NAComputation() = default;
  [[nodiscard]] auto getAtoms() -> decltype(atoms)& {
    return atoms;
  }
  [[nodiscard]] auto getInitialLocations() ->
      decltype(initialLocations)& {
    return initialLocations;
  }
  [[nodiscard]] auto toString() const -> std::string;
  friend auto operator<<(std::ostream& os, const NAComputation& qc)
      -> std::ostream& {
    return os << qc.toString();
  }
  [[nodiscard]] auto validate() const -> bool;
};
} // namespace na
