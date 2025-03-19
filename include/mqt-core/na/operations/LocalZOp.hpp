/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing local Z operations.
 */

#pragma once

#include "na/entities/Atom.hpp"
#include "na/operations/LocalOp.hpp"

#include <string>
#include <vector>

namespace na {
/// Represents a local Z operation in the NAComputation.
class LocalZOp final : public LocalOp {
public:
  /// Creates a new Z operation with the given atoms.
  /// @param atom The atoms the operation is applied to.
  explicit LocalZOp(const std::vector<const Atom*>& atom) : LocalOp(atom, {}) {
    name_ = "z";
  }

  /// Creates a new Z operation with the given atom.
  /// @param atom The atom the operation is applied to.
  explicit LocalZOp(const Atom& atom) : LocalZOp({&atom}) {}
};
} // namespace na
