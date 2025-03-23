/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/operations/LocalOp.hpp"

#include "ir/Definitions.hpp"
#include "na/entities/Atom.hpp"

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

namespace na {
namespace {
auto printParams(const std::vector<qc::fp>& params, std::ostringstream& os)
    -> void {
  if (!params.empty()) {
    for (const auto& p : params) {
      os << p << " ";
    }
  }
}
} // namespace

auto LocalOp::toString() const -> std::string {
  std::ostringstream ss;
  ss << std::setprecision(5) << std::fixed;
  ss << "@+ " << name_ << " ";
  if (atoms_.size() == 1) {
    printParams(params_, ss);
    ss << *atoms_.front();
    return ss.str();
  }
  ss << "[\n";
  for (const auto& atom : atoms_) {
    ss << "    ";
    printParams(params_, ss);
    ss << *atom << "\n";
  }
  ss << "]";
  return ss.str();
}
} // namespace na
