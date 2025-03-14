/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/operations/LocalOp.hpp"

#include "Definitions.hpp"
#include "na/entities/Atom.hpp"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace na {
auto LocalOp::printParams(const std::vector<qc::fp>& params, std::ostream& os)
    -> void {
  if (!params.empty()) {
    for (const auto& p : params) {
      os << p << " ";
    }
  }
}
auto LocalOp::printAtoms(const std::vector<const Atom*>& atoms,
                         std::ostream& os) -> void {
  if (atoms.size() == 1) {
    os << *(atoms.front());
  } else {
    os << "{";
    for (size_t i = 0; i < atoms.size(); ++i) {
      // skip comma in first iteration
      if (i > 0) {
        os << ", ";
      }
      os << *atoms[i];
    }
    os << "}";
  }
}
LocalOp::LocalOp(const std::vector<const Atom*>& atoms,
                 std::vector<qc::fp> params)
    : params_(std::move(params)) {
  std::transform(
      atoms.cbegin(), atoms.cend(), std::back_inserter(atoms_),
      [](const auto* const a) { return std::vector<const Atom*>{a}; });
}
auto LocalOp::toString() const -> std::string {
  std::stringstream ss;
  ss << std::setprecision(5) << std::fixed;
  ss << "@+ " << name_ << " ";
  if (atoms_.size() == 1) {
    printParams(params_, ss);
    printAtoms(atoms_.front(), ss);
    return ss.str();
  }
  ss << "[\n";
  for (const auto& atoms : atoms_) {
    ss << "    ";
    printParams(params_, ss);
    printAtoms(atoms, ss);
    ss << "\n";
  }
  ss << "]";
  return ss.str();
}
} // namespace na
