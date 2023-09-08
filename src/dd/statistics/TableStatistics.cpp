#include "dd/statistics/TableStatistics.hpp"

#include <nlohmann/json.hpp>

namespace dd {

void TableStatistics::trackInsert() noexcept {
  ++inserts;
  ++numEntries;
  peakNumEntries = std::max(peakNumEntries, numEntries);
}

void TableStatistics::reset() noexcept {
  numBuckets = 0U;
  numEntries = 0U;
  collisions = 0U;
  hits = 0U;
  lookups = 0U;
  inserts = 0U;
}

fp TableStatistics::hitRatio() const noexcept {
  if (lookups == 0) {
    return 1.;
  }
  return static_cast<fp>(hits) / static_cast<fp>(lookups);
}

fp TableStatistics::colRatio() const noexcept {
  if (lookups == 0) {
    return 0.;
  }
  return static_cast<fp>(collisions) / static_cast<fp>(lookups);
}

fp TableStatistics::loadFactor() const noexcept {
  if (numBuckets == 0) {
    return 0.;
  }
  return static_cast<fp>(numEntries) / static_cast<fp>(numBuckets);
}

nlohmann::json TableStatistics::json() const {
  nlohmann::json j = Statistics::json();
  j["num_buckets"] = numBuckets;
  j["num_entries"] = numEntries;
  j["peak_num_entries"] = peakNumEntries;
  j["collisions"] = collisions;
  j["hits"] = hits;
  j["lookups"] = lookups;
  j["inserts"] = inserts;
  j["hit_ratio"] = hitRatio();
  j["col_ratio"] = colRatio();
  j["load_factor"] = loadFactor();
  return j;
}

} // namespace dd
