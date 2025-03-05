/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/operations/StoreOp.hpp"

#include <cstddef>
#include <sstream>
#include <string>

namespace na {
auto StoreOp::toString() const -> std::string {
  std::stringstream ss;
  ss << "@+ store";
  if (atoms_.size() == 1) {
    if (targetLocations_) {
      ss << " " << targetLocations_->front();
    }
    ss << " " << *(atoms_.front());
    return ss.str();
  }
  ss << " [\n";
  for (std::size_t i = 0; i < atoms_.size(); ++i) {
    ss << "    ";
    if (targetLocations_) {
      ss << (*targetLocations_)[i] << " ";
    }
    ss << *(atoms_[i]) << "\n";
  }
  ss << "]";
  return ss.str();
}
} // namespace na
