/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Definitions.hpp"
#include "operations/Control.hpp"

#include <cstddef>
#include <functional>
#include <map>

namespace qc {
class Permutation : public std::map<Qubit, Qubit> {
public:
  [[nodiscard]] auto apply(const Controls& controls) const -> Controls;
  [[nodiscard]] auto apply(const Targets& targets) const -> Targets;
  [[nodiscard]] auto apply(Qubit qubit) const -> Qubit;
  [[nodiscard]] auto maxKey() const -> Qubit;
  [[nodiscard]] auto maxValue() const -> Qubit;
};
} // namespace qc

// define hash function for Permutation
template <> struct std::hash<qc::Permutation> {
  std::size_t operator()(const qc::Permutation& p) const noexcept {
    std::size_t seed = 0;
    for (const auto& [k, v] : p) {
      qc::hashCombine(seed, k);
      qc::hashCombine(seed, v);
    }
    return seed;
  }
};
