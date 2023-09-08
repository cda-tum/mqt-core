#include "dd/statistics/MemoryManagerStatistics.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace dd {

std::size_t
MemoryManagerStatistics::getNumAvailableFromChunks() const noexcept {
  return numAllocated - numUsed;
}

std::size_t MemoryManagerStatistics::getTotalNumAvailable() const noexcept {
  return getNumAvailableFromChunks() + numAvailableForReuse;
}

void MemoryManagerStatistics::trackUsedEntries(
    const std::size_t numEntries) noexcept {
  numUsed += numEntries;
  peakNumUsed = std::max(peakNumUsed, numUsed);
}

void MemoryManagerStatistics::trackReusedEntries(
    const std::size_t numEntries) noexcept {
  numUsed += numEntries;
  peakNumUsed = std::max(peakNumUsed, numUsed);
  numAvailableForReuse -= numEntries;
}

void MemoryManagerStatistics::trackReturnedEntry() noexcept {
  ++numAvailableForReuse;
  peakNumAvailableForReuse =
      std::max(peakNumAvailableForReuse, numAvailableForReuse);
  --numUsed;
}

void MemoryManagerStatistics::reset() noexcept {
  numAllocations = 0U;
  numAllocated = 0U;
  numUsed = 0U;
  numAvailableForReuse = 0U;
}

nlohmann::json MemoryManagerStatistics::json() const {
  nlohmann::json j = Statistics::json();
  j["num_allocations"] = numAllocations;
  j["num_allocated"] = numAllocated;
  j["num_used"] = numUsed;
  j["num_available_for_reuse"] = numAvailableForReuse;
  j["peak_num_used"] = peakNumUsed;
  j["peak_num_available_for_reuse"] = peakNumAvailableForReuse;
  return j;
}

} // namespace dd
