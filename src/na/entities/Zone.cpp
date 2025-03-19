/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/entities/Zone.hpp"

#include "na/entities/Location.hpp"

#include <stdexcept>

namespace na {

auto Zone::contains(const Location& location) const -> bool {
  if (!extent_) {
    throw std::runtime_error("Zone's extent is not set.");
  }
  return extent_->minX <= location.x && location.x <= extent_->maxX &&
         extent_->minY <= location.y && location.y <= extent_->maxY;
}
} // namespace na
