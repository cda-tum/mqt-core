#include "dd/RealNumberUniqueTable.hpp"

#include "dd/RealNumber.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace dd {

RealNumberUniqueTable::RealNumberUniqueTable(MemoryManager<RealNumber>& manager,
                                             const std::size_t initialGCLim)
    : memoryManager(&manager), initialGCLimit(initialGCLim) {
  // add 1/2 to the complex table and increase its ref count (so that it is
  // not collected)
  lookup(0.5L)->ref++;
}

fp RealNumberUniqueTable::tolerance() noexcept { return RealNumber::eps; }

void RealNumberUniqueTable::setTolerance(const fp tol) noexcept {
  RealNumber::eps = tol;
}

std::int64_t RealNumberUniqueTable::hash(const fp val) noexcept {
  static constexpr std::int64_t MASK = NBUCKET - 1;
  assert(val >= 0);
  const auto key = static_cast<std::int64_t>(std::nearbyint(val * MASK));
  return std::min<std::int64_t>(key, MASK);
}

RealNumber* RealNumberUniqueTable::lookup(const fp val) {
  assert(!std::isnan(val));
  assert(val >= 0); // required anyway for the hash function
  ++lookups;
  if (RealNumber::approximatelyZero(val)) {
    ++hits;
    return &constants::zero;
  }

  if (RealNumber::approximatelyOne(val)) {
    ++hits;
    return &constants::one;
  }

  if (RealNumber::approximatelyEquals(val, SQRT2_2)) {
    ++hits;
    return &constants::sqrt2over2;
  }
  assert(val - RealNumber::eps >= 0); // should be handle above as special case

  const auto lowerKey = hash(val - RealNumber::eps);
  const auto upperKey = hash(val + RealNumber::eps);

  if (upperKey == lowerKey) {
    ++findOrInserts;
    return findOrInsert(lowerKey, val);
  }

  // code below is to properly handle border cases |----(-|-)----|
  // in case a value close to a border is looked up,
  // only the last entry in the lower bucket and the first entry in the upper
  // bucket need to be checked

  const auto key = hash(val);

  RealNumber* pLower; // NOLINT(cppcoreguidelines-init-variables)
  RealNumber* pUpper; // NOLINT(cppcoreguidelines-init-variables)
  if (lowerKey != key) {
    pLower = tailTable[static_cast<std::size_t>(lowerKey)];
    pUpper = table[static_cast<std::size_t>(key)];
    ++lowerNeighbors;
  } else {
    pLower = tailTable[static_cast<std::size_t>(key)];
    pUpper = table[static_cast<std::size_t>(upperKey)];
    ++upperNeighbors;
  }

  const bool lowerMatchFound =
      (pLower != nullptr &&
       RealNumber::approximatelyEquals(val, pLower->value));
  const bool upperMatchFound =
      (pUpper != nullptr &&
       RealNumber::approximatelyEquals(val, pUpper->value));

  if (lowerMatchFound && upperMatchFound) {
    ++hits;
    const auto diffToLower = std::abs(pLower->value - val);
    const auto diffToUpper = std::abs(pUpper->value - val);
    // val is actually closer to p_lower than to p_upper
    if (diffToLower < diffToUpper) {
      return pLower;
    }
    return pUpper;
  }

  if (lowerMatchFound) {
    ++hits;
    return pLower;
  }

  if (upperMatchFound) {
    ++hits;
    return pUpper;
  }

  // value was not found in the table -> get a new entry and add it to the
  // central bucket
  return insert(key, val);
}

bool RealNumberUniqueTable::possiblyNeedsCollection() const noexcept {
  return memoryManager->getUsedCount() >= gcLimit;
}

std::size_t RealNumberUniqueTable::garbageCollect(const bool force) noexcept {
  gcCalls++;
  // nothing to be done if garbage collection is not forced, and the limit has
  // not been reached, or the current count is minimal (the complex table
  // always contains at least 0.5)
  if ((!force && !possiblyNeedsCollection()) ||
      memoryManager->getUsedCount() <= 1) {
    return 0;
  }

  gcRuns++;
  std::size_t collected = 0;
  std::size_t remaining = 0;
  for (std::size_t key = 0; key < table.size(); ++key) {
    auto* p = table[key];
    RealNumber* lastp = nullptr;
    while (p != nullptr) {
      if (p->ref == 0) {
        auto* next = p->next;
        if (lastp == nullptr) {
          table[key] = next;
        } else {
          lastp->next = next;
        }
        memoryManager->free(p);
        p = next;
        collected++;
      } else {
        lastp = p;
        p = p->next;
        remaining++;
      }
      tailTable[key] = lastp;
    }
  }
  // The garbage collection limit changes dynamically depending on the number
  // of remaining (active) nodes. If it were not changed, garbage collection
  // would run through the complete table on each successive call once the
  // number of remaining entries reaches the garbage collection limit. It is
  // increased whenever the number of remaining entries is rather close to the
  // garbage collection threshold and decreased if the number of remaining
  // entries is much lower than the current limit.
  if (remaining > gcLimit / 10 * 9) {
    gcLimit = remaining + initialGCLimit;
  } else if (remaining < gcLimit / 128) {
    gcLimit /= 2;
  }
  return collected;
}

void RealNumberUniqueTable::clear() noexcept {
  // clear table buckets
  for (auto& bucket : table) {
    bucket = nullptr;
  }
  for (auto& entry : tailTable) {
    entry = nullptr;
  }

  collisions = 0;
  insertCollisions = 0;
  hits = 0;
  findOrInserts = 0;
  lookups = 0;
  inserts = 0;
  lowerNeighbors = 0;
  upperNeighbors = 0;

  gcCalls = 0;
  gcRuns = 0;
  gcLimit = initialGCLimit;
}

void RealNumberUniqueTable::print() const {
  const auto precision = std::cout.precision();
  std::cout.precision(std::numeric_limits<dd::fp>::max_digits10);
  for (std::size_t key = 0; key < table.size(); ++key) {
    auto* p = table[key];
    if (p != nullptr) {
      std::cout << key << ": \n";
    }

    while (p != nullptr) {
      std::cout << "\t\t" << p->value << " "
                << reinterpret_cast<std::uintptr_t>(p) << " " << p->ref << "\n";
      p = p->next;
    }

    if (table[key] != nullptr) {
      std::cout << "\n";
    }
  }
  std::cout.precision(precision);
}

fp RealNumberUniqueTable::hitRatio() const noexcept {
  if (lookups == 0) {
    return 0.0;
  }
  return static_cast<fp>(hits) / static_cast<fp>(lookups);
}

fp RealNumberUniqueTable::colRatio() const noexcept {
  if (lookups == 0) {
    return 0.0;
  }
  return static_cast<fp>(collisions) / static_cast<fp>(lookups);
}

std::map<std::string, std::size_t, std::less<>>
RealNumberUniqueTable::getStatistics() noexcept {
  return {
      {"hits", hits},
      {"collisions", collisions},
      {"lookups", lookups},
      {"inserts", inserts},
      {"insertCollisions", insertCollisions},
      {"findOrInserts", findOrInserts},
      {"upperNeighbors", upperNeighbors},
      {"lowerNeighbors", lowerNeighbors},
      {"gcCalls", gcCalls},
      {"gcRuns", gcRuns},
  };
}

std::ostream& RealNumberUniqueTable::printStatistics(std::ostream& os) const {
  os << "hits: " << hits << ", collisions: " << collisions
     << ", looks: " << lookups << ", inserts: " << inserts
     << ", insertCollisions: " << insertCollisions
     << ", findOrInserts: " << findOrInserts
     << ", upperNeighbors: " << upperNeighbors
     << ", lowerNeighbors: " << lowerNeighbors << ", hitRatio: " << hitRatio()
     << ", colRatio: " << colRatio() << ", gc calls: " << gcCalls
     << ", gc runs: " << gcRuns << "\n";
  return os;
}

std::ostream& RealNumberUniqueTable::printBucketDistribution(std::ostream& os) {
  for (auto* bucket : table) {
    if (bucket == nullptr) {
      os << "0\n";
      continue;
    }
    std::size_t bucketCount = 0;
    while (bucket != nullptr) {
      ++bucketCount;
      bucket = bucket->next;
    }
    os << bucketCount << "\n";
  }
  os << "\n";
  return os;
}

RealNumber* RealNumberUniqueTable::findOrInsert(const std::int64_t key,
                                                const fp val) {
  const fp valTol = val + RealNumber::eps;

  auto* curr = table[static_cast<std::size_t>(key)];
  RealNumber* prev = nullptr;

  while (curr != nullptr && curr->value <= valTol) {
    if (RealNumber::approximatelyEquals(curr->value, val)) {
      // check if val is actually closer to the next element in the list (if
      // there is one)
      if (curr->next != nullptr) {
        const auto& next = curr->next;
        // potential candidate in range
        if (valTol >= next->value) {
          const auto diffToCurr = std::abs(curr->value - val);
          const auto diffToNext = std::abs(next->value - val);
          // val is actually closer to next than to curr
          if (diffToNext < diffToCurr) {
            ++hits;
            return next;
          }
        }
      }
      ++hits;
      return curr;
    }
    ++collisions;
    prev = curr;
    curr = curr->next;
  }

  ++inserts;
  auto* entry = memoryManager->get();
  entry->value = val;

  if (prev == nullptr) {
    // table bucket is empty
    table[static_cast<std::size_t>(key)] = entry;
  } else {
    prev->next = entry;
  }
  entry->next = curr;
  if (curr == nullptr) {
    tailTable[static_cast<std::size_t>(key)] = entry;
  }
  return entry;
}

RealNumber* RealNumberUniqueTable::insert(const std::int64_t key,
                                          const fp val) {
  ++inserts;
  auto* entry = memoryManager->get();
  entry->value = val;

  auto* curr = table[static_cast<std::size_t>(key)];
  RealNumber* prev = nullptr;

  while (curr != nullptr && curr->value <= val) {
    ++insertCollisions;
    prev = curr;
    curr = curr->next;
  }

  if (prev == nullptr) {
    // table bucket is empty
    table[static_cast<std::size_t>(key)] = entry;
  } else {
    prev->next = entry;
  }
  entry->next = curr;
  if (curr == nullptr) {
    tailTable[static_cast<std::size_t>(key)] = entry;
  }
  return entry;
}

} // namespace dd
