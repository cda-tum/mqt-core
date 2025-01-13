/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/statistics/Statistics.hpp"
#include "nlohmann/json_fwd.hpp"

#include <cstddef>

namespace dd {

/**
 * @brief A utility class for storing statistics of a memory manager
 * @tparam T the type of entries managed by the memory manager
 */
template <typename T> struct MemoryManagerStatistics : public Statistics {
  /// The size of a single entry
  std::size_t entrySize = sizeof(T);
  /// The number of allocations performed
  std::size_t numAllocations = 0U;
  /// The number of allocated entries
  std::size_t numAllocated = 0U;
  /// The number of entries currently in use
  std::size_t numUsed = 0U;
  /// The number of entries currently available for reuse
  std::size_t numAvailableForReuse = 0U;
  /// The peak number of entries in use
  std::size_t peakNumUsed = 0U;
  /// The peak number of entries available for reuse
  std::size_t peakNumAvailableForReuse = 0U;

  static constexpr auto ENTRY_MEMORY_MIB =
      static_cast<double>(sizeof(T)) / static_cast<double>(1ULL << 20U);

  /// Get the number of available entries from memory chunks
  [[nodiscard]] std::size_t getNumAvailableFromChunks() const noexcept;

  /// Get the total number of available entries
  [[nodiscard]] std::size_t getTotalNumAvailable() const noexcept;

  /// Get an estimate for ratio of used memory
  [[nodiscard]] double getUsageRatio() const noexcept;

  /// Get an estimate of the total allocated memory in MiB
  [[nodiscard]] double getAllocatedMemoryMiB() const noexcept;

  /// Get an estimate of the total used memory in MiB
  [[nodiscard]] double getUsedMemoryMiB() const noexcept;

  /// Get an estimate for the peak used memory in MiB
  [[nodiscard]] double getPeakUsedMemoryMiB() const noexcept;

  /// Track newly used entries (from chunks)
  void trackUsedEntries(std::size_t numEntries = 1U) noexcept;

  /// Track reused entries (from available list)
  void trackReusedEntries(std::size_t numEntries = 1U) noexcept;

  /// Track a new available entry for reuse
  void trackReturnedEntry() noexcept;

  /// Reset all statistics (except for the peak values)
  void reset() noexcept override;

  /// Get a JSON representation of the statistics
  [[nodiscard]] nlohmann::json json() const override;
};

} // namespace dd
