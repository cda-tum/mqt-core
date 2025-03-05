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
#include "na/entities/Atom.hpp"
#include "na/operations/LocalOp.hpp"

#include <string>
#include <utility>
#include <vector>

namespace na {
class LocalRZOp final : public LocalOp {
public:
  LocalRZOp(std::vector<const Atom*> atom, const qc::fp angle)
      : LocalOp(std::move(atom), {angle}) {
    name_ = "rz";
  }
  LocalRZOp(const Atom& atom, const qc::fp angle) : LocalRZOp({&atom}, angle) {
    name_ = "rz";
  }
};
} // namespace na
