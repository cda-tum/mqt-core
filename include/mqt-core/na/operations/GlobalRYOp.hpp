/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing global RY operations.
 */

#pragma once

#include "Definitions.hpp"
#include "na/entities/Zone.hpp"
#include "na/operations/GlobalOp.hpp"

namespace na {
/// Represents a global RY operation in the NAComputation.
class GlobalRYOp final : public GlobalOp {
public:
  /// Creates a new RY operation in the given zone with the given angle.
  /// @param zone The zone the operation is applied to.
  /// @param angle The angle of the operation.
  GlobalRYOp(const Zone& zone, qc::fp angle) : GlobalOp(zone, {angle}) {
    name_ = "ry";
  }
};
} // namespace na
