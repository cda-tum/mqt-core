/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/statistics/TableStatistics.hpp"

#include <array>
#include <bitset>
#include <cstddef>
#include <functional>

namespace dd {

/**
 * @brief Data structure for caching computed results of unary operations
 * @tparam OperandType type of the operation's operand
 * @tparam ResultType type of the operation's result
 * @tparam NBUCKET number of hash buckets to use (has to be a power of two)
 */
template <class OperandType, class ResultType, std::size_t NBUCKET = 32768>
class UnaryComputeTable {
public:
  /// Default constructor
  UnaryComputeTable() {
    stats.entrySize = sizeof(Entry);
    stats.numBuckets = NBUCKET;
  }

  /// An entry in the compute table
  struct Entry {
    OperandType operand;
    ResultType result;
  };

  /// Bitmask used in the hash function
  static constexpr size_t MASK = NBUCKET - 1;

  /// Get a reference to the underlying table
  [[nodiscard]] const auto& getTable() const { return table; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  /// Compute the hash value for a given operand
  static std::size_t hash(const OperandType& a) {
    return std::hash<OperandType>{}(a)&MASK;
  }

  /**
   * @brief Insert a new entry into the compute table
   * @details Any existing entry for the resulting hash value will be replaced.
   * @param operand The operand
   * @param result The result of the operation
   */
  void insert(const OperandType& operand, const ResultType& result) {
    const auto key = hash(operand);
    if (valid[key]) {
      ++stats.collisions;
    } else {
      stats.trackInsert();
      valid.set(key);
    }
    table[key] = {operand, result};
  }

  /**
   * @brief Look up a result in the compute table
   * @param operand The operand
   * @return A pointer to the result if it is found, otherwise nullptr.
   */
  ResultType* lookup(const OperandType& operand) {
    ResultType* result = nullptr;
    ++stats.lookups;
    const auto key = hash(operand);

    if (!valid[key]) {
      return result;
    }

    auto& entry = table[key];
    if (entry.operand != operand) {
      return result;
    }

    ++stats.hits;
    return &entry.result;
  }

  /**
   * @brief Clear the compute table
   * @details Sets all entries to invalid.
   */
  void clear() { valid.reset(); }

private:
  /// The actual table storing the entries
  std::array<Entry, NBUCKET> table{};
  /// Bitset to mark valid entries
  std::bitset<NBUCKET> valid{};
  /// Statistics of the compute table
  TableStatistics stats{};
};
} // namespace dd
