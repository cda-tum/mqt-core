#include "dd/statistics/MemoryManagerStatistics.hpp"

#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace dd {

template <typename T>
std::size_t
MemoryManagerStatistics<T>::getNumAvailableFromChunks() const noexcept {
  return numAllocated - numUsed;
}

template <typename T>
std::size_t MemoryManagerStatistics<T>::getTotalNumAvailable() const noexcept {
  return getNumAvailableFromChunks() + numAvailableForReuse;
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

template <typename T> nlohmann::json MemoryManagerStatistics<T>::json() const {
  nlohmann::json j = Statistics::json();
  j["num_allocations"] = numAllocations;
  j["num_allocated"] = numAllocated;
  j["memory_allocated_MiB"] = getAllocatedMemoryMiB();
  j["num_used"] = numUsed;
  j["memory_used_MiB"] = getUsedMemoryMiB();
  j["num_available_for_reuse"] = numAvailableForReuse;
  j["total_num_available"] = getTotalNumAvailable();
  j["usage_ratio"] = getUsageRatio();
  j["peak_num_used"] = peakNumUsed;
  j["peak_num_available_for_reuse"] = peakNumAvailableForReuse;
  j["peak_memory_used_MiB"] = getPeakUsedMemoryMiB();
  return j;
}

template struct MemoryManagerStatistics<RealNumber>;
template struct MemoryManagerStatistics<vNode>;
template struct MemoryManagerStatistics<mNode>;
template struct MemoryManagerStatistics<dNode>;
} // namespace dd
