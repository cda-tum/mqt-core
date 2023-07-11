#pragma once

#include "dd/Definitions.hpp"

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <ostream>
#include <string>

namespace dd {
/// \brief A class for storing statistics of a unique table
struct UniqueTableStatistics {
  // statistics on the table performance
  /// The number of collisions
  std::size_t collisions = 0U;
  /// The number of successful lookups
  std::size_t hits = 0U;
  /// The number of lookups
  std::size_t lookups = 0U;
  /// The number of inserts
  std::size_t inserts = 0U;

  // statistics on the entries in the table
  /// The number of entries in the table
  std::size_t entryCount = 0U;
  /// The peak number of entries in the table
  std::size_t peakEntryCount = 0U;
  /**
   * @brief The total number of active entries
   * @details An entry is considered active if it has a non-zero reference count
   */
  std::size_t activeEntryCount = 0U;
  /// The peak number of active entries in the table
  std::size_t peakActiveEntryCount = 0U;

  // statistics on garbage collection
  /// The number of garbage collection calls
  std::size_t gcCalls = 0U;
  /// The number of garbage actual garbage collection runs
  std::size_t gcRuns = 0U;

  /// Track a new insert
  void trackInsert() noexcept;

  /// Track a new active entry
  void trackActiveEntry() noexcept;

  /// Reset all statistics
  void reset() noexcept;

  /**
   * @brief Get the hit ratio of the table.
   * @details The hit ratio is the ratio of lookups that were successful.
   * @returns The hit ratio of the table.
   */
  [[nodiscard]] fp hitRatio() const noexcept {
    if (lookups == 0) {
      return 1.;
    }
    return static_cast<fp>(hits) / static_cast<fp>(lookups);
  }

  /**
   * @brief Get the collision ratio of the table.
   * @details A collision occurs when the hash function maps two different
   * entries to the same bucket. The collision ratio is the ratio of lookups
   * that resulted in a collision.
   * @returns The collision ratio of the table.
   */
  [[nodiscard]] fp colRatio() const noexcept {
    if (lookups == 0) {
      return 0.;
    }
    return static_cast<fp>(collisions) / static_cast<fp>(lookups);
  }

  /// Get a JSON representation of the statistics
  [[nodiscard]] nlohmann::json json() const;
  /// Get a pretty printed string representation of the statistics
  [[nodiscard]] std::string toString() const;
  /**
   * @brief Write a string representation to an output stream
   * @param os The output stream
   * @param stats The statistics
   * @return The output stream
   */
  friend std::ostream& operator<<(std::ostream& os,
                                  const UniqueTableStatistics& stats) {
    os << stats.toString();
    return os;
  }
};

} // namespace dd
