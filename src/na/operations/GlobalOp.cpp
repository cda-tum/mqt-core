/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/operations/GlobalOp.hpp"

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

namespace na {
auto GlobalOp::printParams(const std::vector<qc::fp>& params, std::ostream& os)
    -> void {
  if (!params.empty()) {
    for (const auto& p : params) {
      os << p << " ";
    }
  }
}
auto GlobalOp::toString() const -> std::string {
  std::stringstream ss;
  ss << std::setprecision(5) << std::fixed;
  ss << "@+ " << name_ << " ";
  if (zones_.size() == 1) {
    printParams(params_, ss);
    ss << *zones_.front();
    return ss.str();
  }
  ss << "[\n";
  for (const auto& atom : zones_) {
    ss << "    ";
    printParams(params_, ss);
    ss << *atom << "\n";
  }
  ss << "]";
  return ss.str();
}
} // namespace na
