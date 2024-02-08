#pragma once

#include "DDDefinitions.hpp"
#include "dd/statistics/TableStatistics.hpp"

#include <array>
#include <bitset>
#include <cstddef>
#include <iostream>
#include <utility>

namespace dd {

/// Data structure for caching computed results of unary operations
/// \tparam OperandType type of the operation's operand
/// \tparam ResultType type of the operation's result
/// \tparam NBUCKET number of hash buckets to use (has to be a power of two)
template <class OperandType, class ResultType, std::size_t NBUCKET = 32768>
class UnaryComputeTable {
public:
  UnaryComputeTable() {
    stats.entrySize = sizeof(Entry);
    stats.numBuckets = NBUCKET;
  }

  struct Entry {
    OperandType operand;
    ResultType result;
  };

  static constexpr size_t MASK = NBUCKET - 1;

  /// Get a reference to the table
  [[nodiscard]] const auto& getTable() const { return table; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  static std::size_t hash(const OperandType& a) {
    return std::hash<OperandType>{}(a)&MASK;
  }

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
