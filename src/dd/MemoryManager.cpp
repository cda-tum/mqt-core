#include "dd/MemoryManager.hpp"

#include "dd/ComplexTable.hpp"
#include "dd/Node.hpp"

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

template <typename T> T* MemoryManager<T>::getTemporary() {
  if (entryAvailableForReuse()) {
    return available;
  }

  if (!entryAvailableInChunk()) {
    allocateNewChunk();
  }

  return &(*chunkIt);
}

template <typename T> void MemoryManager<T>::free(T* entry) {
  assert(entry != nullptr);
  assert(entry->ref == 0);
  entry->next = available;
  available = entry;
  ++availableForReuseCount;
  --usedCount;
}

template <typename T>
void MemoryManager<T>::reset(const bool resizeToTotal) noexcept {
  available = nullptr;
  availableForReuseCount = 0U;

  chunks.erase(chunks.begin() + 1, chunks.end());
  for (auto& entry : chunks[0]) {
    entry.ref = 0U;
  }
  if (resizeToTotal) {
    chunks[0].resize(allocationCount);
  }

  chunkIt = chunks[0].begin();
  chunkEndIt = chunks[0].end();
  allocationCount = chunks[0].size();
  usedCount = 0U;
}

template <typename T>
T* MemoryManager<T>::getEntryFromAvailableList() noexcept {
  assert(entryAvailableForReuse());
  auto* entry = available;
  available = available->next;
  --availableForReuseCount;
  // Reclaimed entries might have a non-zero reference count
  entry->ref = 0;
  return entry;
}

template <typename T> void MemoryManager<T>::allocateNewChunk() {
  assert(!entryAvailableInChunk());
  const auto newChunkSize = static_cast<std::size_t>(
      GROWTH_FACTOR * static_cast<double>(chunks.back().size()));
  chunks.emplace_back(newChunkSize);
  chunkIt = chunks.back().begin();
  chunkEndIt = chunks.back().end();
  allocationCount += newChunkSize;
}

template <typename T> T* MemoryManager<T>::getEntryFromChunk() noexcept {
  assert(!entryAvailableForReuse());
  assert(entryAvailableInChunk());
  auto* entry = &(*chunkIt);
  ++chunkIt;
  ++usedCount;
  return entry;
}

template class MemoryManager<CTEntry>;
template class MemoryManager<vNode>;
template class MemoryManager<mNode>;
template class MemoryManager<dNode>;

} // namespace dd
