/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/operations/LocalOp.hpp"

#include <iomanip>
#include <sstream>
#include <string>

namespace na {
auto LocalOp::toString() const -> std::string {
  std::stringstream ss;
  ss << std::setprecision(5) << std::fixed;
  ss << "@+ " << name;
  if (atoms.size() == 1) {
    if (!params.empty()) {
      for (const auto& p : params) {
        ss << " " << p;
      }
    }
    ss << " " << *(atoms.front());
  } else {
    ss << " [\n";
    for (const auto* const atom : atoms) {
      ss << "    ";
      if (!params.empty()) {
        for (const auto& p : params) {
          ss << p << " ";
        }
      }
      ss << *atom << "\n";
    }
    ss << "]";
  }
  return ss.str();
}
} // namespace na
