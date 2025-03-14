/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing local RZ operations.
 */

#pragma once

#include "na/entities/Atom.hpp"
#include "na/operations/LocalOp.hpp"

#include <array>
#include <string>
#include <vector>

namespace na {
/// Represents a local RZ operation in the NAComputation.
class LocalCZOp final : public LocalOp {
public:
  /// Creates a new RZ operation with the given atoms and angle.
  /// @param atoms The atoms the operation is applied to.
  explicit LocalCZOp(const std::vector<std::array<const Atom*, 2>>& atoms)
      : LocalOp(atoms, {}) {
    name_ = "cz";
  }

  /// Creates a new RZ operation with the given atom and angle.
  /// @param atom The atom the operation is applied to.
  explicit LocalCZOp(const std::array<const Atom*, 2>& atom)
      : LocalCZOp(std::vector{atom}) {}

  /// Creates a new RZ operation with the given atom and angle.
  /// @param atom1 The atom the operation is applied to.
  /// @param atom2 The atom the operation is applied to.
  explicit LocalCZOp(const Atom& atom1, const Atom& atom2)
      : LocalCZOp(std::vector{std::array{&atom1, &atom2}}) {}
};
} // namespace na
