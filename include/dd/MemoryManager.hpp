#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/statistics/MemoryManagerStatistics.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace dd {

/**
 * @brief A memory manager for objects of type T.
 * @details The class manages a collection of objects of type T. The objects are
 * stored in contiguous chunks of memory. The manager supports reclaiming
 * objects that are no longer in use. This is done by maintaining a linked list
 * of available objects. When an object is no longer in use, it is added to the
 * list. When a new object is requested, the first object from the list is
 * returned. If the list is empty, an object from the current chunk is returned.
 * If the current chunk is full, a new chunk is allocated. The size of chunks
 * grows exponentially according to a growth factor.
 * @note The main purpose of this class is to reduce the number of memory
 * allocations and deallocations. This is achieved by allocating a large number
 * of objects at once and reusing them. This is especially useful for objects
 * that are frequently created and destroyed, such as decision diagram nodes,
 * edge weights, etc.
 * @tparam T The type of objects to manage.
 */
template <typename T> class MemoryManager {
  static_assert(std::is_same_v<decltype(T::next), T*>,
                "T must have a `next` member of type T*");
  static_assert(std::is_same_v<decltype(T::ref), RefCount>,
                "T must have a `ref` member of type RefCount");

public:
  /**
   * @brief The number of initially allocated entries.
   * @details The number of initially allocated entries is the number of entries
   * that are allocated as a chunk when the manager is created. Increasing this
   * number reduces the number of allocations, but increases the memory usage.
   */
  static constexpr std::size_t INITIAL_ALLOCATION_SIZE = 2048U;

  /**
   * @brief The growth factor for table entry allocation.
   * @details The growth factor is used to determine the number of entries that
   * are allocated when the manager runs out of entries. Per default, the number
   * of entries is doubled. Increasing this number reduces the number of memory
   * allocations, but increases the memory usage.
   */
  static constexpr double GROWTH_FACTOR = 2U;

  /**
   * @brief Construct a new MemoryManager object
   * @param initialAllocationSize The initial number of entries to allocate
   */
  explicit MemoryManager(
      const std::size_t initialAllocationSize = INITIAL_ALLOCATION_SIZE)
      : chunks(1, std::vector<T>(initialAllocationSize)),
        chunkIt(chunks[0].begin()), chunkEndIt(chunks[0].end()) {
    stats.numAllocations = 1U;
    stats.numAllocated = initialAllocationSize;
  }

  /// default destructor
  ~MemoryManager() = default;

  /**
   * @brief Get an entry from the manager.
   * @details If an entry is available for reuse, it is returned. Otherwise, an
   * entry from the pre-allocated chunks is returned. If no entry is available,
   * a new chunk is allocated.
   * @return A pointer to an entry.
   */
  [[nodiscard]] T* get();

  /**
   * @brief Get a pair of entries from the manager.
   * @return A pair of pointers to entries.
   * @see get()
   * @note This method assumes that there is an even number of entries available
   * from the manager. If this is not the case, the behavior is undefined.
   */
  [[nodiscard]] std::pair<T*, T*> getPair();

  /**
   * @brief Return an entry to the manager.
   * @details The entry is added to the list of available entries. The entry
   * must not be used after it has been returned to the manager. Entries should
   * have a reference count of 0 when they are returned to the manager. If not,
   * this indicates a reference counting error.
   * @param entry A pointer to an entry that is no longer in use.
   */
  void returnEntry(T* entry) noexcept;

  /**
   * @brief Reset the manager.
   * @details Drops all but the first chunk and resets the reference counts of
   * all entries to 0. If `resizeToTotal` is set to true, the first chunk is
   * resized to the total number of entries. This increases memory locality
   * and reduces the number of allocations when the manager is used again.
   * However, it might also require a huge contiguous block of memory to be
   * allocated.
   * @param resizeToTotal If set to true, the first chunk is resized to the
   * total number of entries.
   */
  void reset(bool resizeToTotal = false) noexcept;

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

private:
  /**
   * @brief Check whether an entry is available for reuse
   * @return true if an entry is available for reuse, false otherwise
   */
  [[nodiscard]] bool entryAvailableForReuse() const noexcept {
    return available != nullptr;
  }

  /**
   * @brief Get an entry from the list of available entries
   * @return A pointer to an entry ready for reuse
   */
  [[nodiscard]] T* getEntryFromAvailableList() noexcept;

  /**
   * @brief Check whether an entry is available in the current chunk
   * @return true if an entry is available in the current chunk, false
   * otherwise
   */
  [[nodiscard]] bool entryAvailableInChunk() const noexcept {
    return chunkIt != chunkEndIt;
  }

  /// Allocate a new chunk of memory
  void allocateNewChunk();

  /**
   * @brief Get an entry from the current chunk
   * @return A pointer to an entry from the current chunk
   */
  [[nodiscard]] T* getEntryFromChunk() noexcept;

  /**
   * @brief A linked list of entries that are available for (re-)use
   * @details The MemoryManager maintains a linked list of entries that are
   * available for (re-)use. This list is implemented as a singly linked list
   * using the `next` member of the entries. The `available` member points to
   * the first entry in the list. If the list is empty, `available` is
   * `nullptr`.
   */
  T* available{};

  /**
   * @brief The storage for the entries
   * @details The MemoryManager maintains a vector of chunks. Each chunk is a
   * vector of entries. Entries in a chunk are allocated contiguously.
   */
  std::vector<std::vector<T>> chunks;

  /**
   * @brief Iterator to the next available entry in the current chunk
   * @details This iterator points to the next available entry in the current
   * chunk. If the current chunk is full, it points to the end of the chunk.
   */
  typename std::vector<T>::iterator chunkIt;

  /**
   * @brief Iterator to the end of the current chunk
   * @details This iterator points to the end of the current chunk. It is used
   * to determine whether the current chunk is full.
   */
  typename std::vector<T>::iterator chunkEndIt;

  /// Memory manager statistics
  MemoryManagerStatistics<T> stats{};
};

} // namespace dd
