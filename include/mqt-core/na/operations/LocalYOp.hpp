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

#include <string>
#include <utility>
#include <vector>

namespace na {
/// Represents a local RZ operation in the NAComputation.
class LocalYOp final : public LocalOp {
public:
  /// Creates a new RZ operation with the given atoms and angle.
  /// @param atom The atoms the operation is applied to.
  explicit LocalYOp(std::vector<const Atom*> atom)
      : LocalOp(std::move(atom), {}) {
    name_ = "y";
  }

  /// Creates a new RZ operation with the given atom and angle.
  /// @param atom The atom the operation is applied to.
  explicit LocalYOp(const Atom& atom) : LocalYOp({&atom}) {}
};
} // namespace na
