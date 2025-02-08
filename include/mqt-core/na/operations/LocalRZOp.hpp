/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "na/operations/LocalOp.hpp"

#include <string>

namespace na {
class LocalRZOp final : public LocalOp {
public:
  LocalRZOp(qc::fp angle, const Atom* atom) : LocalOp({angle}, atom) {
    name = "rz";
  }
};
} // namespace na
