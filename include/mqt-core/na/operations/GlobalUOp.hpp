/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing global U3 operations.
 */

#pragma once

#include "Definitions.hpp"
#include "LocalOp.hpp"
#include "LocalRYOp.hpp"
#include "na/entities/Atom.hpp"
#include "na/entities/Location.hpp"
#include "na/entities/Zone.hpp"
#include "na/operations/GlobalOp.hpp"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace na {
/// Represents a global U3 operation in the NAComputation.
class GlobalUOp final : public GlobalOp {
public:
  /// Creates a new RY operation in the given zone with the given angle.
  /// @param zone The zone the operation is applied to.
  /// @param angle The angle of the operation.
  GlobalUOp(const Zone& zone, qc::fp theta, qc::fp phi, qc::fp lambda)
      : GlobalOp(zone, {theta, phi, lambda}) {
    name_ = "u";
  }

  /// Returns a local representation of the operation.
  /// @param atomsLocations The locations of the atoms.
  [[nodiscard]] auto
  toLocal(const std::unordered_map<const Atom*, Location>& atomsLocations,
          const double /* unused */) const
      -> std::unique_ptr<LocalOp> override {
    // make sure atoms are added always in the same order based on their
    // location
    std::map<Location, const Atom*> sortedAtoms;
    for (const auto& [atom, loc] : atomsLocations) {
      sortedAtoms.emplace(loc, atom);
    }
    std::vector<const Atom*> affectedAtoms;
    for (const auto& [loc, atom] : sortedAtoms) {
      if (zone_->contains(loc)) {
        affectedAtoms.emplace_back(atom);
      }
    }
    return std::make_unique<LocalRYOp>(affectedAtoms, params_.front());
  }
};
} // namespace na
