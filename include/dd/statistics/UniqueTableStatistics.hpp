#pragma once

#include "dd/statistics/TableStatistics.hpp"

namespace dd {
/// \brief A class for storing statistics of a unique table
struct UniqueTableStatistics : public TableStatistics {
  /**
   * @brief The total number of active entries
   * @details An entry is considered active if it has a non-zero reference count
   */
  std::size_t numActiveEntries = 0U;
  /// The peak number of active entries in the table
  std::size_t peakNumActiveEntries = 0U;
  /// The number of garbage collection runs
  std::size_t gcRuns = 0U;

  /// Track a new active entry
  void trackActiveEntry() noexcept;

  /// Reset all statistics (except for the peak values)
  void reset() noexcept override;

  /// Get a JSON representation of the statistics
  [[nodiscard]] nlohmann::json json() const override;
};

} // namespace dd
