/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "na/entities/Zone.hpp"
#include "na/operations/GlobalOp.hpp"

namespace na {
/// Represents a global cz operation in the NA computation.
class GlobalCZOp final : public GlobalOp {
public:
  /// Creates a new cz operation in the given zone.
  /// @param zone The zone the operation is applied to.
  explicit GlobalCZOp(const Zone& zone) : GlobalOp(zone, {}) { name_ = "cz"; }
};
} // namespace na
