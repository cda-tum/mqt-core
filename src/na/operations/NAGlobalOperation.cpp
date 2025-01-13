/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/operations/NAGlobalOperation.hpp"

#include <ios>
#include <sstream>
#include <string>

namespace na {
auto NAGlobalOperation::toString() const -> std::string {
  std::stringstream ss;
  ss << type;
  if (!params.empty()) {
    ss << "(";
    for (const auto& p : params) {
      ss << p << ", ";
    }
    ss.seekp(-2, std::ios_base::end);
    ss << ")";
  }
  ss << ";\n";
  return ss.str();
}
} // namespace na
