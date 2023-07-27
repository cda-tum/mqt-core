#pragma once

#include "DDDefinitions.hpp"

#include <array>
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
  UnaryComputeTable() = default;

  struct Entry {
    OperandType operand;
    ResultType result;
  };

  static constexpr size_t MASK = NBUCKET - 1;

  // access functions
  [[nodiscard]] const auto& getTable() const { return table; }

  static std::size_t hash(const OperandType& a) {
    return std::hash<OperandType>{}(a)&MASK;
  }

  void insert(const OperandType& operand, const ResultType& result) {
    const auto key = hash(operand);
    table[key] = {operand, result};
    ++count;
  }

  ResultType* lookup(const OperandType& operand) {
    ResultType* result = nullptr;
    lookups++;
    const auto key = hash(operand);
    auto& entry = table[key];
    if (entry.operand != operand) {
      return result;
    }

    hits++;
    return &entry.result;
  }

  void clear() {
    if (count > 0) {
      std::fill(table.begin(), table.end(), Entry{});
      count = 0;
    }
    hits = 0;
    lookups = 0;
  }

  [[nodiscard]] fp hitRatio() const {
    return static_cast<fp>(hits) / static_cast<fp>(lookups);
  }

  std::ostream& printStatistics(std::ostream& os = std::cout) {
    os << "hits: " << hits << ", looks: " << lookups
       << ", ratio: " << hitRatio() << std::endl;
    return os;
  }

private:
  std::array<Entry, NBUCKET> table{};
  // compute table lookup statistics
  std::size_t hits = 0;
  std::size_t lookups = 0;
  std::size_t count = 0;
};
} // namespace dd
