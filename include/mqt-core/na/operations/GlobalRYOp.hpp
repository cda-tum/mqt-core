/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Definitions.hpp"
#include "na/entities/Zone.hpp"
#include "na/operations/GlobalOp.hpp"

namespace na {
class GlobalRYOp final : public GlobalOp {
public:
  GlobalRYOp(const Zone& zone, qc::fp angle) : GlobalOp(zone, {angle}) {
    name_ = "ry";
  }
};
} // namespace na
