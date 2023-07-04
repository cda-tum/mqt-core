#include "dd/ComplexTable.hpp"

#include <algorithm>
#include <iomanip>
#include <stdexcept>

namespace dd {

static constexpr std::size_t LSB = static_cast<std::uintptr_t>(1U);

CTEntry* CTEntry::getAlignedPointer(const Entry* e) {
  return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(e) & ~LSB);
}

CTEntry* CTEntry::getNegativePointer(const Entry* e) {
  return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(e) | LSB);
}

bool CTEntry::exactlyZero(const Entry* e) { return (e == &zero); }

bool CTEntry::exactlyOne(const Entry* e) { return (e == &one); }

CTEntry* CTEntry::flipPointerSign(const Entry* e) {
  if (exactlyZero(e)) {
    return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(e));
  }
  return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(e) ^ LSB);
}

bool CTEntry::isNegativePointer(const Entry* e) {
  return (reinterpret_cast<std::uintptr_t>(e) & LSB) != 0U;
}

fp CTEntry::val(const Entry* e) {
  if (isNegativePointer(e)) {
    return -getAlignedPointer(e)->value;
  }
  return e->value;
}

RefCount CTEntry::ref(const Entry* e) {
  if (isNegativePointer(e)) {
    return -getAlignedPointer(e)->refCount;
  }
  return e->refCount;
}

bool CTEntry::approximatelyEquals(const fp left, const fp right) {
  // NOLINTNEXTLINE(clang-diagnostic-float-equal)
  return left == right || std::abs(left - right) <= TOLERANCE;
}

bool CTEntry::approximatelyEquals(const Entry* left, const Entry* right) {
  return left == right || approximatelyEquals(val(left), val(right));
}

bool CTEntry::approximatelyZero(const fp e) { return std::abs(e) <= TOLERANCE; }

bool CTEntry::approximatelyZero(const Entry* e) {
  return e == &zero || approximatelyZero(val(e));
}

bool CTEntry::approximatelyOne(const fp e) {
  return approximatelyEquals(e, 1.0);
}

bool CTEntry::approximatelyOne(const Entry* e) {
  return e == &one || approximatelyOne(val(e));
}

void CTEntry::writeBinary(const Entry* e, std::ostream& os) {
  const auto temp = val(e);
  os.write(reinterpret_cast<const char*>(&temp), sizeof(decltype(temp)));
}

ComplexTable::ComplexTable(const std::size_t initialAllocSize,
                           const std::size_t growthFact,
                           const std::size_t initialGCLim)
    : initialAllocationSize(initialAllocSize), growthFactor(growthFact),
      initialGCLimit(initialGCLim) {
  // add 1/2 to the complex table and increase its ref count (so that it is
  // not collected)
  lookup(0.5L)->refCount++;
}

CTEntry* ComplexTable::lookup(const fp val) {
  assert(!std::isnan(val));
  assert(val >= 0); // required anyway for the hash function
  ++lookups;
  if (Entry::approximatelyZero(val)) {
    ++hits;
    return &zero;
  }

  if (Entry::approximatelyOne(val)) {
    ++hits;
    return &one;
  }

  if (Entry::approximatelyEquals(val, SQRT2_2)) {
    ++hits;
    return &sqrt2over2;
  }

  assert(val - TOLERANCE >= 0); // should be handle above as special case

  const auto lowerKey = hash(val - TOLERANCE);
  const auto upperKey = hash(val + TOLERANCE);

  if (upperKey == lowerKey) {
    ++findOrInserts;
    return findOrInsert(lowerKey, val);
  }

  // code below is to properly handle border cases |----(-|-)----|
  // in case a value close to a border is looked up,
  // only the last entry in the lower bucket and the first entry in the upper
  // bucket need to be checked

  const auto key = hash(val);

  Entry* pLower; // NOLINT(cppcoreguidelines-init-variables)
  Entry* pUpper; // NOLINT(cppcoreguidelines-init-variables)
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
      (pLower != nullptr && Entry::approximatelyEquals(val, pLower->value));
  const bool upperMatchFound =
      (pUpper != nullptr && Entry::approximatelyEquals(val, pUpper->value));

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

CTEntry* ComplexTable::getEntry() {
  // an entry is available on the stack
  if (!availableEmpty()) {
    auto* entry = available;
    available = entry->next;
    // returned entries could have a ref count != 0
    entry->refCount = 0;
    return entry;
  }

  // new chunk has to be allocated
  if (chunkIt == chunkEndIt) {
    chunks.emplace_back(allocationSize);
    allocations += allocationSize;
    allocationSize *= growthFactor;
    chunkID++;
    chunkIt = chunks[chunkID].begin();
    chunkEndIt = chunks[chunkID].end();
  }

  auto* entry = &(*chunkIt);
  ++chunkIt;
  return entry;
}

void ComplexTable::returnEntry(Entry* entry) {
  entry->next = available;
  available = entry;
}

void ComplexTable::incRef(Entry* entry) {
  // get valid pointer
  auto* entryPtr = Entry::getAlignedPointer(entry);

  if (entryPtr == nullptr) {
    return;
  }

  if (isStaticEntry(entryPtr)) {
    return;
  }

  if (entryPtr->refCount == std::numeric_limits<RefCount>::max()) {
    std::clog << "[WARN] MAXREFCNT reached for " << entryPtr->value
              << ". Number will never be collected.\n";
    return;
  }

  // increase reference count
  entryPtr->refCount++;
}

void ComplexTable::decRef(Entry* entry) {
  // get valid pointer
  auto* entryPtr = Entry::getAlignedPointer(entry);

  if (entryPtr == nullptr) {
    return;
  }

  if (isStaticEntry(entryPtr)) {
    return;
  }

  if (entryPtr->refCount == std::numeric_limits<RefCount>::max()) {
    return;
  }

  if (entryPtr->refCount == 0) {
    throw std::runtime_error("In ComplexTable: RefCount of entry " +
                             std::to_string(entryPtr->value) +
                             " is zero before decrement");
  }

  // decrease reference count
  entryPtr->refCount--;
}

bool ComplexTable::possiblyNeedsCollection() const { return count >= gcLimit; }
std::size_t ComplexTable::garbageCollect(const bool force) {
  gcCalls++;
  // nothing to be done if garbage collection is not forced, and the limit has
  // not been reached, or the current count is minimal (the complex table
  // always contains at least 0.5)
  if ((!force && !possiblyNeedsCollection()) || count <= 1) {
    return 0;
  }

  gcRuns++;
  std::size_t collected = 0;
  std::size_t remaining = 0;
  for (std::size_t key = 0; key < table.size(); ++key) {
    Entry* p = table[key];
    Entry* lastp = nullptr;
    while (p != nullptr) {
      if (p->refCount == 0) {
        Entry* next = p->next;
        if (lastp == nullptr) {
          table[key] = next;
        } else {
          lastp->next = next;
        }
        returnEntry(p);
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
  count = remaining;
  return collected;
}

void ComplexTable::clear() {
  // clear table buckets
  for (auto& bucket : table) {
    bucket = nullptr;
  }
  for (auto& entry : tailTable) {
    entry = nullptr;
  }

  // clear available stack
  available = nullptr;

  // release memory of all but the first chunk
  // it could be desirable to keep the memory for later use
  while (chunkID > 0) {
    chunks.pop_back();
    chunkID--;
  }
  // restore initial chunk setting
  chunkIt = chunks[0].begin();
  chunkEndIt = chunks[0].end();
  allocationSize = initialAllocationSize * growthFactor;
  allocations = initialAllocationSize;

  for (auto& entry : chunks[0]) {
    entry.refCount = 0;
  }

  count = 0;
  peakCount = 0;

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

void ComplexTable::print() {
  const auto precision = std::cout.precision();
  std::cout.precision(std::numeric_limits<dd::fp>::max_digits10);
  for (std::size_t key = 0; key < table.size(); ++key) {
    auto* p = table[key];
    if (p != nullptr) {
      std::cout << key << ": \n";
    }

    while (p != nullptr) {
      std::cout << "\t\t" << p->value << " "
                << reinterpret_cast<std::uintptr_t>(p) << " " << p->refCount
                << "\n";
      p = p->next;
    }

    if (table[key] != nullptr) {
      std::cout << "\n";
    }
  }
  std::cout.precision(precision);
}

fp ComplexTable::hitRatio() const {
  return static_cast<fp>(hits) / static_cast<fp>(lookups);
}

fp ComplexTable::colRatio() const {
  return static_cast<fp>(collisions) / static_cast<fp>(lookups);
}

std::map<std::string, std::size_t, std::less<>> ComplexTable::getStatistics() {
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

std::ostream& ComplexTable::printStatistics(std::ostream& os) const {
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

std::ostream& ComplexTable::printBucketDistribution(std::ostream& os) {
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

ComplexTable::Entry* ComplexTable::findOrInsert(const std::int64_t key,
                                                const fp val) {
  const fp valTol = val + TOLERANCE;

  auto* curr = table[static_cast<std::size_t>(key)];
  Entry* prev = nullptr;

  while (curr != nullptr && curr->value <= valTol) {
    if (Entry::approximatelyEquals(curr->value, val)) {
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
  auto* entry = getEntry();
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
  count++;
  peakCount = std::max(peakCount, count);
  return entry;
}

ComplexTable::Entry* ComplexTable::insert(const std::int64_t key,
                                          const fp val) {
  ++inserts;
  auto* entry = getEntry();
  entry->value = val;

  auto* curr = table[static_cast<std::size_t>(key)];
  Entry* prev = nullptr;

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
  count++;
  peakCount = std::max(peakCount, count);
  return entry;
}

} // namespace dd
