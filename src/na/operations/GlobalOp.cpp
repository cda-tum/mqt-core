/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/operations/GlobalRYOp.hpp"

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

namespace na {
auto GlobalOp::toString() const -> std::string {
  std::stringstream ss;
  ss << std::setprecision(5) << std::fixed;
  ss << "@+ " << name;
  if (!params.empty()) {
    for (const auto& p : params) {
      ss << " " << p;
    }
  }
  ss << " " << zone;
  return ss.str();
}
} // namespace na
