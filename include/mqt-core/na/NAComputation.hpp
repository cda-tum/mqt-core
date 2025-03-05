/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "na/entities/Atom.hpp"
#include "na/entities/Location.hpp"
#include "na/entities/Zone.hpp"
#include "na/operations/Op.hpp"

#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace na {
class NAComputation final {
protected:
  std::vector<std::unique_ptr<Op>> operations_;
  std::vector<std::unique_ptr<Atom>> atoms_;
  std::vector<std::unique_ptr<Zone>> zones_;
  std::unordered_map<const Atom*, Location> initialLocations_;

public:
  [[nodiscard]] auto begin() -> decltype(operations_.begin()) {
    return operations_.begin();
  }
  [[nodiscard]] auto begin() const -> decltype(operations_.begin()) {
    return operations_.begin();
  }
  [[nodiscard]] auto end() -> decltype(operations_.end()) {
    return operations_.end();
  }
  [[nodiscard]] auto end() const -> decltype(operations_.end()) {
    return operations_.end();
  }
  [[nodiscard]] auto size() const -> std::size_t { return operations_.size(); }
  [[nodiscard]] auto operator[](std::size_t i) -> Op& {
    return *operations_[i];
  }
  [[nodiscard]] auto operator[](std::size_t i) const -> const Op& {
    return *operations_[i];
  }
  auto clear() -> void { operations_.clear(); }
  [[nodiscard]] auto getAtomsSize() const -> std::size_t {
    return atoms_.size();
  }
  /// Returns the atoms used in the NAComputation.
  [[nodiscard]] auto getAtoms() const -> const decltype(atoms_)& {
    return atoms_;
  }
  /// Returns the zones used in global operations within the NAComputation.
  [[nodiscard]] auto getZones() const -> const decltype(zones_)& {
    return zones_;
  }
  /// Returns the initial locations of the atoms.
  [[nodiscard]] auto getInitialLocations() const -> const
      decltype(initialLocations_)& {
    return initialLocations_;
  }
  /// Returns the location of the given atom after the given operation.
  [[nodiscard]] auto getLocationOfAtomAfterOperation(const Atom& atom,
                                                     const Op& op) const
      -> Location;
  /// Emplaces a new atom with the given name and returns a reference to the
  /// newly created atom.
  auto emplaceBackAtom(std::string name) -> const Atom& {
    return *atoms_.emplace_back(std::make_unique<Atom>(std::move(name)));
  }
  /// Emplaces a new operation of type T with the given operation and returns a
  /// reference to the newly created operation.
  auto emplaceBackZone(std::string name) -> const Zone& {
    return *zones_.emplace_back(std::make_unique<Zone>(std::move(name)));
  }
  /// Emplaces a new initial location for the given atom with the given location
  /// and returns a reference to the newly created location.
  auto emplaceInitialLocation(const Atom* atom, const Location& loc)
      -> const Location& {
    return initialLocations_.emplace(atom, loc).first->second;
  }
  /// Emplaces a new initial location for the given atom with the given
  /// arguments and returns a reference to the newly created location.
  template <typename... Args>
  auto emplaceInitialLocation(const Atom* atom, Args&&... loc)
      -> const Location& {
    return initialLocations_.emplace(atom, Location{std::forward<Args>(loc)...})
        .first->second;
  }
  /// Emplaces a new operation of type T with the given operation and returns a
  /// reference to the newly created operation.
  template <class T> auto emplaceBack(T&& op) -> const Op& {
    return *std::vector<std::unique_ptr<Op>>::emplace_back(
        std::make_unique<T>(std::forward<T>(op)));
  }
  /// Emplaces a new operation of type T with the given arguments and returns a
  /// reference to the newly created operation.
  template <class T, typename... Args>
  auto emplaceBack(Args&&... args) -> const Op& {
    return *std::vector<std::unique_ptr<Op>>::emplace_back(
        std::make_unique<T>(std::forward<Args>(args)...));
  }
  /// Returns a string representation of the NAComputation.
  [[nodiscard]] auto toString() const -> std::string;
  /// Outputs the NAComputation to the given output stream, i.e., the string
  /// returned by toString().
  friend auto operator<<(std::ostream& os, const NAComputation& qc)
      -> std::ostream& {
    return os << qc.toString();
  }
  /// Validates the NAComputation and checks whether all AOD constraints are
  /// fulfilled.
  /// I.e.,
  /// - each atom is loaded before it is moved
  /// - the relative order of loaded atoms is preserved
  /// - each atom is loaded before it is stored
  /// - each atom is stored before it is loaded (again)
  /// @returns a pair of a boolean indicating whether the NAComputation is valid
  /// and a string containing the error message if the NAComputation is invalid.
  [[nodiscard]] auto validate() const -> std::pair<bool, std::string>;
};
} // namespace na
