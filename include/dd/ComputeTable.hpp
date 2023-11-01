#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/statistics/TableStatistics.hpp"

#include <array>
#include <bitset>
#include <cstddef>
#include <iostream>
#include <utility>

namespace dd {

/// Data structure for caching computed results
/// \tparam LeftOperandType type of the operation's left operand
/// \tparam RightOperandType type of the operation's right operand
/// \tparam ResultType type of the operation's result
/// \tparam NBUCKET number of hash buckets to use (has to be a power of two)
template <class LeftOperandType, class RightOperandType, class ResultType,
          std::size_t NBUCKET = 16384>
class ComputeTable {
public:
  ComputeTable() {
    stats.entrySize = sizeof(Entry);
    stats.numBuckets = NBUCKET;
  }

  struct Entry {
    LeftOperandType leftOperand;
    RightOperandType rightOperand;
    ResultType result;
  };

  static constexpr std::size_t MASK = NBUCKET - 1;

  static std::size_t hash(const LeftOperandType& leftOperand,
                          const RightOperandType& rightOperand) {
    const auto h1 = std::hash<LeftOperandType>{}(leftOperand);
    const auto h2 = std::hash<RightOperandType>{}(rightOperand);
    const auto hash = qc::combineHash(h1, h2);
    return hash & MASK;
  }

  /// Get a reference to the table
  [[nodiscard]] const auto& getTable() const { return table; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  void insert(const LeftOperandType& leftOperand,
              const RightOperandType& rightOperand, const ResultType& result) {
    const auto key = hash(leftOperand, rightOperand);
    if (valid[key]) {
      ++stats.collisions;
    } else {
      stats.trackInsert();
      valid.set(key);
    }
    table[key] = {leftOperand, rightOperand, result};
  }

  ResultType* lookup(const LeftOperandType& leftOperand,
                     const RightOperandType& rightOperand,
                     [[maybe_unused]] const bool useDensityMatrix = false) {
    ResultType* result = nullptr;
    ++stats.lookups;
    const auto key = hash(leftOperand, rightOperand);
    if (!valid[key]) {
      return result;
    }

    auto& entry = table[key];
    if (entry.leftOperand != leftOperand) {
      return result;
    }
    if (entry.rightOperand != rightOperand) {
      return result;
    }

    if constexpr (std::is_same_v<RightOperandType, dEdge>) {
      // Since density matrices are reduced representations of matrices, a
      // density matrix may not be returned when a matrix is required and vice
      // versa
      if (!dNode::isTerminal(entry.result.p) &&
          dNode::isDensityMatrixNode(entry.result.p->flags) !=
              useDensityMatrix) {
        return result;
      }
    }
    ++stats.hits;
    return &entry.result;
  }

  void clear() {
    valid.reset();
    stats.reset();
  }

  std::ostream& printStatistics(std::ostream& os = std::cout) {
    return os << stats;
  }

private:
  std::array<Entry, NBUCKET> table{};
  std::bitset<NBUCKET> valid{};
  TableStatistics stats{};
};
} // namespace dd
