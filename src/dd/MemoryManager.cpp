#include "dd/MemoryManager.hpp"

#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

#include <cassert>

namespace dd {

template <typename T> T* MemoryManager<T>::get() {
  if (entryAvailableForReuse()) {
    return getEntryFromAvailableList();
  }

  if (!entryAvailableInChunk()) {
    allocateNewChunk();
  }

  return getEntryFromChunk();
}

template <typename T> std::pair<T*, T*> MemoryManager<T>::getPair() {
  if (entryAvailableForReuse()) {
    auto* r = available;
    assert(r->next != nullptr && "At least two entries must be available");
    auto* i = available->next;
    available = i->next;
    stats.trackReusedEntries(2U);
    return {r, i};
  }

  if (!entryAvailableInChunk()) {
    allocateNewChunk();
  }

  auto* r = &(*chunkIt);
  ++chunkIt;
  assert(chunkIt != chunkEndIt && "At least two entries must be available");
  auto* i = &(*chunkIt);
  ++chunkIt;
  stats.trackUsedEntries(2U);
  return {r, i};
}

template <typename T> void MemoryManager<T>::returnEntry(T* entry) noexcept {
  assert(entry != nullptr);
  assert(entry->ref == 0);
  entry->next = available;
  available = entry;
  stats.trackReturnedEntry();
}

template <typename T>
void MemoryManager<T>::reset(const bool resizeToTotal) noexcept {
  available = nullptr;

  auto numAllocations = stats.numAllocations;
  chunks.resize(1U);
  if (resizeToTotal) {
    chunks[0].resize(stats.numAllocated);
    ++numAllocations;
  }

  chunkIt = chunks[0].begin();
  chunkEndIt = chunks[0].end();

  stats.reset();
  stats.numAllocations = numAllocations;
  stats.numAllocated = chunks[0].size();
}

template <typename T>
T* MemoryManager<T>::getEntryFromAvailableList() noexcept {
  assert(entryAvailableForReuse());
  auto* entry = available;
  available = available->next;
  stats.trackReusedEntries();
  return entry;
}

template <typename T> void MemoryManager<T>::allocateNewChunk() {
  assert(!entryAvailableInChunk());
  const auto newChunkSize = static_cast<std::size_t>(
      GROWTH_FACTOR * static_cast<double>(chunks.back().size()));
  chunks.emplace_back(newChunkSize);
  chunkIt = chunks.back().begin();
  chunkEndIt = chunks.back().end();
  ++stats.numAllocations;
  stats.numAllocated += newChunkSize;
}

template <typename T> T* MemoryManager<T>::getEntryFromChunk() noexcept {
  assert(!entryAvailableForReuse());
  assert(entryAvailableInChunk());
  auto* entry = &(*chunkIt);
  ++chunkIt;
  stats.trackUsedEntries();
  return entry;
}

template class MemoryManager<RealNumber>;
template class MemoryManager<vNode>;
template class MemoryManager<mNode>;
template class MemoryManager<dNode>;

} // namespace dd
