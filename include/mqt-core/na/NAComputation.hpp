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
#include "entities/Zone.hpp"
#include "operations/Op.hpp"

#include <iterator>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace na {
class NAComputation final {
protected:
  std::vector<Atom> atoms;
  std::vector<Zone> zones;
  std::unordered_map<const Atom*, Location> initialLocations;
  std::vector<std::unique_ptr<Op>> operations;

public:
  NAComputation() = default;
  NAComputation(const NAComputation& qc) = default;
  NAComputation(NAComputation&& qc) noexcept = default;
  NAComputation& operator=(const NAComputation& qc) = default;
  NAComputation& operator=(NAComputation&& qc) noexcept = default;
  [[nodiscard]] auto getAtoms() -> decltype(atoms)& { return atoms; }
  [[nodiscard]] auto getAtoms() const -> const decltype(atoms)& {
    return atoms;
  }
  [[nodiscard]] auto getZones() -> decltype(zones)& { return zones; }
  [[nodiscard]] auto getZones() const -> const decltype(zones)& {
    return zones;
  }
  [[nodiscard]] auto getInitialLocations() -> decltype(initialLocations)& {
    return initialLocations;
  }
  [[nodiscard]] auto getInitialLocations() const -> const
      decltype(initialLocations)& {
    return initialLocations;
  }
  template <class T> auto emplaceBack(T&& op) -> std::unique_ptr<Op>& {
    return operations.emplace_back(std::make_unique<T>(std::forward<T>(op)));
  }
  template <class T, typename... Args>
  auto emplaceBack(Args&&... args) -> std::unique_ptr<Op>& {
    return operations.emplace_back(
        std::make_unique<T>(std::forward<Args>(args)...));
  }
  [[nodiscard]] auto begin() -> decltype(operations)::iterator {
    return operations.begin();
  }
  [[nodiscard]] auto begin() const -> decltype(operations)::const_iterator {
    return operations.begin();
  }
  [[nodiscard]] auto end() -> decltype(operations)::iterator {
    return operations.end();
  }
  [[nodiscard]] auto end() const -> decltype(operations)::const_iterator {
    return operations.end();
  }
  auto clear() -> void { operations.clear(); }
  [[nodiscard]] auto empty() const -> bool { return operations.empty(); }
  [[nodiscard]] auto size() const -> std::size_t { return operations.size(); }
  [[nodiscard]] auto operator[](std::size_t i) -> Op& { return *operations[i]; }
  [[nodiscard]] auto operator[](std::size_t i) const -> const Op& {
    return *operations[i];
  }
  [[nodiscard]] auto toString() const -> std::string;
  friend auto operator<<(std::ostream& os, const NAComputation& qc)
      -> std::ostream& {
    return os << qc.toString();
  }
  [[nodiscard]] auto validate() const -> bool;
};
} // namespace na
