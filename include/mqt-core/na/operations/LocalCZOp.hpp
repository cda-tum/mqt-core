/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing local CZ operations.
 */

#pragma once

#include "na/entities/Atom.hpp"
#include "na/operations/LocalOp.hpp"

#include <array>
#include <string>
#include <vector>

namespace na {
/// Represents a local CZ operation in the NAComputation.
class LocalCZOp final : public LocalOp {
public:
  /// Creates a new CZ operation with the given atom pairs.
  /// @param atoms The atom pairs the operation is applied to.
  explicit LocalCZOp(const std::vector<std::array<const Atom*, 2>>& atoms)
      : LocalOp(atoms, {}) {
    name_ = "cz";
  }

  /// Creates a new CZ operation with the given pair of atoms.
  /// @param atom The pair of atoms the operation is applied to.
  explicit LocalCZOp(const std::array<const Atom*, 2>& atom)
      : LocalCZOp(std::vector{atom}) {}

  /// Creates a new CZ operation with the given atom pair.
  /// @param atom1 The first atom of the pair the operation is applied to.
  /// @param atom2 The second atom of the pair the operation is applied to.
  explicit LocalCZOp(const Atom& atom1, const Atom& atom2)
      : LocalCZOp(std::vector{std::array{&atom1, &atom2}}) {}
};
} // namespace na
