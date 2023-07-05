#include "dd/ComplexCache.hpp"

#include <cassert>

namespace dd {

Complex ComplexCache::getCachedComplex() {
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

Complex ComplexCache::getTemporaryComplex() {
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

void ComplexCache::returnToCache(Complex& c) noexcept {
  assert(count >= 2);
  assert(c != Complex::zero);
  assert(c != Complex::one);
  assert(c.r->ref == 0);
  assert(c.i->ref == 0);
  c.i->next = available;
  c.r->next = c.i;
  available = c.r;
  count -= 2;
}

void ComplexCache::clear() noexcept {
  // clear available stack
  available = nullptr;

  // release memory of all but the first chunk
  // it could be desirable to keep the memory for later use
  // or, alternatively, to resize the first chunk to the current allocation size
  // to get better cache locality.
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
}
} // namespace dd
