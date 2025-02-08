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
class GlobalRYOp final : public GlobalOp {
public:
  GlobalRYOp(qc::fp angle, const Zone* zone) : GlobalOp({angle}, zone) {
    name = "ry";
  }
};
} // namespace na
