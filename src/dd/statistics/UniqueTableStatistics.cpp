#include "dd/statistics/UniqueTableStatistics.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace dd {

void UniqueTableStatistics::trackActiveEntry() noexcept {
  ++numActiveEntries;
  peakNumActiveEntries = std::max(peakNumActiveEntries, numActiveEntries);
}

void UniqueTableStatistics::reset() noexcept {
  TableStatistics::reset();
  numActiveEntries = 0U;
  gcCalls = 0U;
  gcRuns = 0U;
}

nlohmann::json UniqueTableStatistics::json() const {
  nlohmann::json j = TableStatistics::json();
  j["num_active_entries"] = numActiveEntries;
  j["peak_num_active_entries"] = peakNumActiveEntries;
  j["gc_calls"] = gcCalls;
  j["gc_runs"] = gcRuns;
  return j;
}
} // namespace dd
