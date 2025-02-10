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
#include "operations/MoveOp.hpp"
#include "operations/Op.hpp"

#include <iterator>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace na {
class NAComputation final : protected std::vector<std::unique_ptr<Op>> {
protected:
  std::vector<std::unique_ptr<Atom>> atoms;
  std::vector<std::unique_ptr<Zone>> zones;
  std::unordered_map<const Atom*, Location> initialLocations;

public:
  using std::vector<std::unique_ptr<Op>>::begin;
  using std::vector<std::unique_ptr<Op>>::clear;
  using std::vector<std::unique_ptr<Op>>::end;
  using std::vector<std::unique_ptr<Op>>::size;
  using std::vector<std::unique_ptr<Op>>::operator[];
  NAComputation() = default;
  NAComputation(const NAComputation& qc) = default;
  NAComputation(NAComputation&& qc) noexcept = default;
  NAComputation& operator=(const NAComputation& qc) = default;
  NAComputation& operator=(NAComputation&& qc) noexcept = default;
  [[nodiscard]] auto getAtomsSize() const -> std::size_t {
    return atoms.size();
  }
  [[nodiscard]] auto getAtoms() const -> const decltype(atoms)& {
    return atoms;
  }
  [[nodiscard]] auto
  getLocationOfAtomAfterOperation(const std::unique_ptr<Atom>& atom,
                                  const std::unique_ptr<Op>& op) const
      -> Location {
    return getLocationOfAtomAfterOperation(atom.get(), op.get());
  }
  [[nodiscard]] auto
  getLocationOfAtomAfterOperation(const std::unique_ptr<Atom>& atom,
                                  const Op* op) const -> Location {
    return getLocationOfAtomAfterOperation(atom.get(), op);
  }
  [[nodiscard]] auto getLocationOfAtomAfterOperation(
      const Atom* atom, const std::unique_ptr<Op>& op) const -> Location {
    return getLocationOfAtomAfterOperation(atom, op.get());
  }
  [[nodiscard]] auto getLocationOfAtomAfterOperation(const Atom* atom,
                                                     const Op* op) const
      -> Location;
  auto emplaceBackAtom(std::string name) -> const Atom* {
    return atoms.emplace_back(std::make_unique<Atom>(std::move(name))).get();
  }
  auto emplaceBackZone(std::string name) -> const Zone* {
    return zones.emplace_back(std::make_unique<Zone>(std::move(name))).get();
  }
  auto emplaceInitialLocation(const Atom* atom, const Location& loc) -> void {
    initialLocations.emplace(atom, loc);
  }
  template <typename... Args>
  auto emplaceInitialLocation(const Atom* atom, Args&&... loc) -> void {
    initialLocations.emplace(atom, Location(std::forward<Args>(loc)...));
  }
  template <class T> auto emplaceBack(T&& op) -> const Op* {
    return std::vector<std::unique_ptr<Op>>::emplace_back(
               std::make_unique<T>(std::forward<T>(op)))
        .get();
  }
  template <class T, typename... Args>
  auto emplaceBack(Args&&... args) -> const Op* {
    return std::vector<std::unique_ptr<Op>>::emplace_back(
               std::make_unique<T>(std::forward<Args>(args)...))
        .get();
  }
  [[nodiscard]] auto toString() const -> std::string;
  friend auto operator<<(std::ostream& os, const NAComputation& qc)
      -> std::ostream& {
    return os << qc.toString();
  }
  [[nodiscard]] auto validate() const -> bool;
};
} // namespace na
