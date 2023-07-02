#pragma once

#include "Complex.hpp"
#include "ComplexTable.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

namespace dd {

class ComplexCache {
  static constexpr std::size_t INITIAL_ALLOCATION_SIZE = 2048U;
  static constexpr std::size_t GROWTH_FACTOR = 2U;

public:
  explicit ComplexCache(
      const std::size_t initialAllocSize = INITIAL_ALLOCATION_SIZE,
      const std::size_t growthFact = GROWTH_FACTOR)
      : initialAllocationSize(initialAllocSize), growthFactor(growthFact) {}

  ~ComplexCache() = default;

  // access functions
  [[nodiscard]] std::size_t getCount() const { return count; }
  [[nodiscard]] std::size_t getPeakCount() const { return peakCount; }
  [[nodiscard]] std::size_t getAllocations() const { return allocations; }
  [[nodiscard]] std::size_t getGrowthFactor() const { return growthFactor; }

  [[nodiscard]] Complex getCachedComplex() {
    // an entry is available on the stack
    if (available != nullptr) {
      assert(available->next != nullptr);
      auto entry = Complex{available, available->next};
      available = entry.i->next;
      count += 2;
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

    Complex c{};
    c.r = &(*chunkIt);
    ++chunkIt;
    c.i = &(*chunkIt);
    ++chunkIt;
    count += 2;
    return c;
  }

  [[nodiscard]] Complex getTemporaryComplex() {
    // an entry is available on the stack
    if (available != nullptr) {
      assert(available->next != nullptr);
      return {available, available->next};
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
    return {&(*chunkIt), &(*(chunkIt + 1))};
  }

  void returnToCache(Complex& c) {
    assert(count >= 2);
    assert(c != Complex::zero);
    assert(c != Complex::one);
    assert(c.r->refCount == 0);
    assert(c.i->refCount == 0);
    c.i->next = available;
    c.r->next = c.i;
    available = c.r;
    count -= 2;
  }

  void clear() {
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

    count = 0;
    peakCount = 0;
  };

private:
  CTEntry* available{};
  std::size_t initialAllocationSize;
  std::size_t growthFactor;
  std::vector<std::vector<CTEntry>> chunks{
      1U, std::vector<CTEntry>(initialAllocationSize)};
  std::size_t chunkID{};
  typename std::vector<CTEntry>::iterator chunkIt{chunks.front().begin()};
  typename std::vector<CTEntry>::iterator chunkEndIt{chunks.front().end()};
  std::size_t allocationSize{initialAllocationSize * growthFactor};

  std::size_t allocations = initialAllocationSize;
  std::size_t count = 0;
  std::size_t peakCount = 0;
};
} // namespace dd
