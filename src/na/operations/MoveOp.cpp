/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/operations/MoveOp.hpp"

#include <cstddef>
#include <sstream>
#include <string>

namespace na {
auto MoveOp::toString() const -> std::string {
  std::stringstream ss;
  ss << "@+ move";
  if (atoms.size() == 1) {
    ss << " " << targetLocations.front() << " " << *(atoms.front());
  } else {
    ss << " [\n";
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      ss << "    " << targetLocations[i] << " " << *(atoms[i]) << "\n";
    }
    ss << "]";
  }
  return ss.str();
}
} // namespace na
