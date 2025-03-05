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
#include <ios>
#include <sstream>
#include <string>

namespace na {
auto LocalOp::toString() const -> std::string {
  std::stringstream ss;
  ss << std::setprecision(5) << std::fixed;
  ss << "@+ " << name_;
  if (atoms_.size() == 1) {
    if (!params_.empty()) {
      for (const auto& p : params_) {
        ss << " " << p;
      }
    }
    ss << " " << *(atoms_.front());
    return ss.str();
  }
  ss << " [\n";
  for (const auto* const atom : atoms_) {
    ss << "    ";
    if (!params_.empty()) {
      for (const auto& p : params_) {
        ss << p << " ";
      }
    }
    ss << *atom << "\n";
  }
  ss << "]";
  return ss.str();
}
} // namespace na
