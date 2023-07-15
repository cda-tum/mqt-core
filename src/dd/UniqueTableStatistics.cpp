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
  nlohmann::json j{};
  auto& perf = j["table_performance"];
  perf["collisions"] = collisions;
  perf["hits"] = hits;
  perf["lookups"] = lookups;
  perf["inserts"] = inserts;
  perf["hit_ratio"] = hitRatio();
  perf["col_ratio"] = colRatio();

  auto& entry = j["entry_statistics"];
  entry["entry_count"] = entryCount;
  entry["peak_entry_count"] = peakEntryCount;
  entry["active_entry_count"] = activeEntryCount;
  entry["peak_active_entry_count"] = peakActiveEntryCount;

  auto& garbage = j["garbage_collection_statistics"];
  garbage["calls"] = gcCalls;
  garbage["runs"] = gcRuns;

  return j;
}

std::string UniqueTableStatistics::toString() const { return json().dump(2U); }
} // namespace dd
