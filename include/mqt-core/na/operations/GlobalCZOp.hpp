/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "na/operations/GlobalOp.hpp"

namespace na {
class GlobalCZOp final : public GlobalOp {
public:
  explicit GlobalCZOp(const Zone* zone) : GlobalOp(zone) { name = "cz"; }
};
} // namespace na
