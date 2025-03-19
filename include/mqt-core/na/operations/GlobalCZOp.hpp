/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing global CZ operations.
 */

#pragma once

#include "LocalCZOp.hpp"
#include "LocalOp.hpp"
#include "na/entities/Atom.hpp"
#include "na/entities/Location.hpp"
#include "na/entities/Zone.hpp"
#include "na/operations/GlobalOp.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace na {
/// Represents a global CZ operation in the NAComputation.
class GlobalCZOp final : public GlobalOp {
public:
  /// Creates a new CZ operation in the given zone.
  /// @param zone The zone the operation is applied to.
  explicit GlobalCZOp(const Zone& zone) : GlobalOp(zone, {}) { name_ = "cz"; }

  /// Creates a new CZ operation in the given zones.
  /// @param zones The zones the operation is applied to.
  explicit GlobalCZOp(const std::vector<const Zone*>& zones)
      : GlobalOp(zones, {}) {
    name_ = "cz";
  }

  /// Returns a local representation of the operation.
  /// @param atomsLocations The locations of the atoms.
  /// @param rydbergRange The range of the Rydberg interaction.
  [[nodiscard]] auto
  toLocal(const std::unordered_map<const Atom*, Location>& atomsLocations,
          const double rydbergRange) const
      -> std::unique_ptr<LocalOp> override {
    // use a map here with the location as keys to ensure a deterministic order
    // of the atoms
    std::map<Location, const Atom*> affectedAtoms;
    for (const auto& [atom, loc] : atomsLocations) {
      if (std::any_of(zones_.cbegin(), zones_.cend(), [&loc](const Zone* zone) {
            return zone->contains(loc);
          })) {
        affectedAtoms.emplace(loc, atom);
      }
    }
    std::vector<std::array<const Atom*, 2>> atomPairs;
    for (auto it1 = affectedAtoms.cbegin(); it1 != affectedAtoms.cend();
         ++it1) {
      const auto& [loc1, atom1] = *it1;
      for (auto it2 = std::next(it1); it2 != affectedAtoms.cend(); ++it2) {
        const auto& [loc2, atom2] = *it2;
        if (Location::distance(loc1, loc2) <= rydbergRange) {
          atomPairs.emplace_back(std::array{atom1, atom2});
        }
      }
    }
    return std::make_unique<LocalCZOp>(atomPairs);
  }
};
} // namespace na
