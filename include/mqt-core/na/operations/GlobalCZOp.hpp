/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing global CZ operations.
 */

#pragma once

#include "na/entities/Zone.hpp"
#include "na/operations/GlobalOp.hpp"

namespace na {
/// Represents a global CZ operation in the NAComputation.
class GlobalCZOp final : public GlobalOp {
public:
  /// Creates a new CZ operation in the given zone.
  /// @param zone The zone the operation is applied to.
  explicit GlobalCZOp(const Zone& zone) : GlobalOp(zone, {}) { name_ = "cz"; }
};
} // namespace na
