/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/statistics/MemoryManagerStatistics.hpp"

#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"
#include "dd/statistics/Statistics.hpp"

#include <algorithm>
#include <cstddef>
#include <nlohmann/json.hpp>

namespace dd {

template <typename T>
std::size_t
MemoryManagerStatistics<T>::getNumAvailableFromChunks() const noexcept {
  return getTotalNumAvailable() - numAvailableForReuse;
}

template <typename T>
std::size_t MemoryManagerStatistics<T>::getTotalNumAvailable() const noexcept {
  return numAllocated - numUsed;
}

template <typename T>
double MemoryManagerStatistics<T>::getUsageRatio() const noexcept {
  return static_cast<double>(numUsed) / static_cast<double>(numAllocated);
}

template <typename T>
double MemoryManagerStatistics<T>::getAllocatedMemoryMiB() const noexcept {
  return static_cast<double>(numAllocated) * ENTRY_MEMORY_MIB;
}

template <typename T>
double MemoryManagerStatistics<T>::getUsedMemoryMiB() const noexcept {
  return static_cast<double>(numUsed) * ENTRY_MEMORY_MIB;
}

template <typename T>
double MemoryManagerStatistics<T>::getPeakUsedMemoryMiB() const noexcept {
  return static_cast<double>(peakNumUsed) * ENTRY_MEMORY_MIB;
}

template <typename T>
void MemoryManagerStatistics<T>::trackUsedEntries(
    const std::size_t numEntries) noexcept {
  numUsed += numEntries;
  peakNumUsed = std::max(peakNumUsed, numUsed);
}

template <typename T>
void MemoryManagerStatistics<T>::trackReusedEntries(
    const std::size_t numEntries) noexcept {
  numUsed += numEntries;
  peakNumUsed = std::max(peakNumUsed, numUsed);
  numAvailableForReuse -= numEntries;
}

template <typename T>
void MemoryManagerStatistics<T>::trackReturnedEntry() noexcept {
  ++numAvailableForReuse;
  peakNumAvailableForReuse =
      std::max(peakNumAvailableForReuse, numAvailableForReuse);
  --numUsed;
}

template <typename T> void MemoryManagerStatistics<T>::reset() noexcept {
  numAllocations = 0U;
  numAllocated = 0U;
  numUsed = 0U;
  numAvailableForReuse = 0U;
}

template <typename T>
nlohmann::basic_json<> MemoryManagerStatistics<T>::json() const {
  if (peakNumUsed == 0) {
    return "unused";
  }

  auto j = Statistics::json();
  j["memory_allocated_MiB"] = getAllocatedMemoryMiB();
  j["memory_used_MiB"] = getUsedMemoryMiB();
  j["memory_used_MiB_peak"] = getPeakUsedMemoryMiB();
  j["num_allocated"] = numAllocated;
  j["num_allocations"] = numAllocations;
  j["num_available_for_reuse"] = numAvailableForReuse;
  j["num_available_for_reuse_peak"] = peakNumAvailableForReuse;
  j["num_available_from_chunks"] = getNumAvailableFromChunks();
  j["num_available_total"] = getTotalNumAvailable();
  j["num_used"] = numUsed;
  j["num_used_peak"] = peakNumUsed;
  j["usage_ratio"] = getUsageRatio();
  return j;
}

template struct MemoryManagerStatistics<RealNumber>;
template struct MemoryManagerStatistics<vNode>;
template struct MemoryManagerStatistics<mNode>;
template struct MemoryManagerStatistics<dNode>;
} // namespace dd
