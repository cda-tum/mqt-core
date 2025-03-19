/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing local X operations.
 */

#pragma once

#include "na/entities/Atom.hpp"
#include "na/operations/LocalOp.hpp"

#include <string>
#include <vector>

namespace na {
/// Represents a local X operation in the NAComputation.
class LocalXOp final : public LocalOp {
public:
  /// Creates a new X operation with the given atoms.
  /// @param atom The atoms the operation is applied to.
  explicit LocalXOp(const std::vector<const Atom*>& atom) : LocalOp(atom, {}) {
    name_ = "x";
  }

  /// Creates a new X operation with the given atom.
  /// @param atom The atom the operation is applied to.
  explicit LocalXOp(const Atom& atom) : LocalXOp({&atom}) {}
};
} // namespace na
