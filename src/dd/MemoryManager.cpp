#include "dd/MemoryManager.hpp"

#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

#include <cassert>

namespace dd {

template <typename T> T* MemoryManager<T>::get() {
  if (entryAvailableForReuse()) {
    if constexpr (std::is_same_v<T, mNode>) {
      std::cout << "Reusing " << std::hex
                << reinterpret_cast<std::size_t>(available) << std::dec << "\n";
    }
    return getEntryFromAvailableList();
  }

  if (!entryAvailableInChunk()) {
    allocateNewChunk();
  }

  if constexpr (std::is_same_v<T, mNode>) {
    std::cout << "Allocating " << std::hex
              << reinterpret_cast<std::size_t>(&(*chunkIt)) << std::dec << "\n";
  }
  return getEntryFromChunk();
}

template <typename T> std::pair<T*, T*> MemoryManager<T>::getPair() {
  if (entryAvailableForReuse()) {
    auto* r = available;
    assert(r->next != nullptr && "At least two entries must be available");
    auto* i = available->next;
    available = i->next;
    usedCount += 2U;
    availableForReuseCount -= 2U;
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
  usedCount += 2U;
  return {r, i};
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

template <typename T> std::pair<T*, T*> MemoryManager<T>::getTemporaryPair() {
  if (entryAvailableForReuse()) {
    assert(available->next != nullptr &&
           "At least two entries must be available");
    return {available, available->next};
  }

  if (!entryAvailableInChunk()) {
    allocateNewChunk();
  }

  assert(chunkIt + 1 != chunkEndIt && "At least two entries must be available");
  return {&(*chunkIt), &(*(chunkIt + 1))};
}

template <typename T> void MemoryManager<T>::returnEntry(T* entry) noexcept {
  assert(entry != nullptr);
  assert(entry->ref == 0);
  entry->next = available;
  available = entry;
  ++availableForReuseCount;
  --usedCount;
  if constexpr (std::is_same_v<T, mNode>) {
    std::cout << "Returning " << std::hex
              << reinterpret_cast<std::size_t>(entry) << std::dec << "\n";
  }
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
  ++usedCount;
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

template class MemoryManager<RealNumber>;
template class MemoryManager<vNode>;
template class MemoryManager<mNode>;
template class MemoryManager<dNode>;

} // namespace dd
