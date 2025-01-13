/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/statistics/TableStatistics.hpp"

#include <array>
#include <bitset>
#include <cstddef>
#include <functional>
#include <vector>

namespace dd {

/// Data structure for caching computed results of noise operations
/// \tparam OperandType type of the operation's operand
/// \tparam ResultType type of the operation's result
/// \tparam NBUCKET number of hash buckets to use (has to be a power of two)
template <class OperandType, class ResultType, std::size_t NBUCKET = 32768>
class DensityNoiseTable { // todo: Inherit from UnaryComputerTable
public:
  DensityNoiseTable() {
    stats.entrySize = sizeof(Entry);
    stats.numBuckets = NBUCKET;
  }

  struct Entry {
    OperandType operand;
    ResultType result;
    std::vector<dd::Qubit> usedQubits;
  };

  static constexpr size_t MASK = NBUCKET - 1;

  /// Get a reference to the table
  [[nodiscard]] const auto& getTable() const { return table; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  static std::size_t hash(const OperandType& a,
                          const std::vector<Qubit>& usedQubits) {
    std::size_t i = 0;
    for (const auto qubit : usedQubits) {
      i = (i << 3U) + i * static_cast<std::size_t>(qubit) +
          static_cast<std::size_t>(qubit);
    }
    return (std::hash<OperandType>{}(a) + i) & MASK;
  }

  void insert(const OperandType& operand, const ResultType& result,
              const std::vector<Qubit>& usedQubits) {
    const auto key = hash(operand, usedQubits);
    if (valid[key]) {
      ++stats.collisions;
    } else {
      stats.trackInsert();
      valid.set(key);
    }
    table[key] = {operand, result, usedQubits};
  }

  ResultType lookup(const OperandType& operand,
                    const std::vector<Qubit>& usedQubits) {
    ResultType result{};
    ++stats.lookups;
    const auto key = hash(operand, usedQubits);

    if (!valid[key]) {
      return result;
    }

    auto& entry = table[key];
    if (entry.operand != operand) {
      return result;
    }
    if (entry.usedQubits != usedQubits) {
      return result;
    }
    ++stats.hits;
    return entry.result;
  }

  void clear() {
    valid.reset();
    stats.reset();
  }

private:
  std::array<Entry, NBUCKET> table{};
  std::bitset<NBUCKET> valid{};
  TableStatistics stats{};
};
} // namespace dd
