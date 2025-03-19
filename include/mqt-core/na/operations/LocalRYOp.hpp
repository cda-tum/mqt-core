/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing local RY operations.
 */

#pragma once

#include "Definitions.hpp"
#include "na/entities/Atom.hpp"
#include "na/operations/LocalOp.hpp"

#include <string>
#include <vector>

namespace na {
/// Represents a local RY operation in the NAComputation.
class LocalRYOp final : public LocalOp {
public:
  /// Creates a new RY operation with the given atoms and angle.
  /// @param atom The atoms the operation is applied to.
  /// @param angle The angle of the operation.
  LocalRYOp(const std::vector<const Atom*>& atom, const qc::fp angle)
      : LocalOp(atom, {angle}) {
    name_ = "ry";
  }

  /// Creates a new RY operation with the given atom and angle.
  /// @param atom The atom the operation is applied to.
  /// @param angle The angle of the operation.
  LocalRYOp(const Atom& atom, const qc::fp angle) : LocalRYOp({&atom}, angle) {}
};
} // namespace na
