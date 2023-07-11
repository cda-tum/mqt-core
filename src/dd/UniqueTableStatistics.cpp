#include "dd/UniqueTableStatistics.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace dd {
void UniqueTableStatistics::trackInsert() noexcept {
  ++inserts;
  ++entryCount;
  peakEntryCount = std::max(peakEntryCount, entryCount);
}

void UniqueTableStatistics::trackActiveEntry() noexcept {
  ++activeEntryCount;
  peakActiveEntryCount = std::max(peakActiveEntryCount, activeEntryCount);
}

void UniqueTableStatistics::reset() noexcept {
  collisions = 0U;
  hits = 0U;
  lookups = 0U;
  inserts = 0U;
  entryCount = 0U;
  peakEntryCount = 0U;
  activeEntryCount = 0U;
  peakActiveEntryCount = 0U;
  gcCalls = 0U;
  gcRuns = 0U;
}

nlohmann::json UniqueTableStatistics::json() const {
  return nlohmann::json{
      {"table_performance",
       {"collisions", collisions},
       {"hits", hits},
       {"lookups", lookups},
       {"inserts", inserts},
       {"hit_ratio", hitRatio()},
       {"col_ratio", colRatio()}},
      {"entry_statistics",
       {"entry_count", entryCount},
       {"peak_entry_count", peakActiveEntryCount},
       {"active_entry_count", activeEntryCount},
       {"peak_active_entry_count", peakActiveEntryCount}},
      {"garbage_collection_statistics", {"calls", gcCalls}, {"runs", gcRuns}}};
}

std::string UniqueTableStatistics::toString() const { return json().dump(2U); }
} // namespace dd
