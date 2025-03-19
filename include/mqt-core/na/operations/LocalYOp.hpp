/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing local Y operations.
 */

#pragma once

#include "na/entities/Atom.hpp"
#include "na/operations/LocalOp.hpp"

#include <string>
#include <vector>

namespace na {
/// Represents a local Y operation in the NAComputation.
class LocalYOp final : public LocalOp {
public:
  /// Creates a new Y operation with the given atoms.
  /// @param atom The atoms the operation is applied to.
  explicit LocalYOp(const std::vector<const Atom*>& atom) : LocalOp(atom, {}) {
    name_ = "y";
  }

  /// Creates a new Y operation with the given atom.
  /// @param atom The atom the operation is applied to.
  explicit LocalYOp(const Atom& atom) : LocalYOp({&atom}) {}
};
} // namespace na
