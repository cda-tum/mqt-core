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
#include "na/operations/Op.hpp"

#include <utility>
#include <vector>

namespace na {
/// Represents a shuttling operation in the NA computation.
/// @details A shuttling operation is the super class for all shuttling related
/// operations, i.e., load, store, and move operations.
class ShuttlingOp : public Op {
protected:
  std::vector<const Atom*> atoms_;
  /// Creates a new shuttling operation with the given atoms.
  /// @param atoms The atoms the operation is applied to.
  explicit ShuttlingOp(std::vector<const Atom*> atoms)
      : atoms_(std::move(atoms)) {}

public:
  ShuttlingOp() = delete;
  /// Returns the atoms the operation is applied to.
  /// @return The atoms the operation is applied to.
  [[nodiscard]] auto getAtoms() const -> const decltype(atoms_)& {
    return atoms_;
  }
  /// Returns true if the shuttling operation has target locations set.
  /// @return True if the shuttling operation has target locations set, false
  /// otherwise.
  [[nodiscard]] virtual auto hasTargetLocations() const -> bool = 0;
  /// Returns the target locations of the shuttling operation.
  /// @return The target locations of the shuttling operation.
  [[nodiscard]] virtual auto getTargetLocations() const
      -> const std::vector<Location>& = 0;
};
} // namespace na
