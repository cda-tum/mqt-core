/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing global RZ operations.
 */

#pragma once

#include "Definitions.hpp"
#include "LocalOp.hpp"
#include "LocalRZOp.hpp"
#include "na/entities/Atom.hpp"
#include "na/entities/Location.hpp"
#include "na/entities/Zone.hpp"
#include "na/operations/GlobalOp.hpp"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace na {
/// Represents a global RZ operation in the NAComputation.
class GlobalRZOp final : public GlobalOp {
public:
  /// Creates a new RZ operation in the given zone with the given angle.
  /// @param zone The zone the operation is applied to.
  /// @param angle The angle of the operation.
  GlobalRZOp(const Zone& zone, qc::fp angle) : GlobalOp(zone, {angle}) {
    name_ = "rz";
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
    return std::make_unique<LocalRZOp>(affectedAtoms, params_.front());
  }
};
} // namespace na
