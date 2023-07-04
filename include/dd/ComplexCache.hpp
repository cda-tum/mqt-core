#pragma once

#include "Complex.hpp"

#include <cstddef>
#include <vector>

namespace dd {
/// A class for managing a cache of Complex numbers
class ComplexCache {
  /// @see ComplexTable::INITIAL_ALLOCATION_SIZE
  static constexpr std::size_t INITIAL_ALLOCATION_SIZE = 2048U;
  /// @see ComplexTable::GROWTH_FACTOR
  static constexpr std::size_t GROWTH_FACTOR = 2U;

public:
  /**
   * @brief Construct a new ComplexCache object
   * @param initialAllocSize The initial allocation size
   * @param growthFact The growth factor used for the allocation size
   * @see ComplexTable::INITIAL_ALLOCATION_SIZE
   * @see ComplexTable::GROWTH_FACTOR
   */
  explicit ComplexCache(
      const std::size_t initialAllocSize = INITIAL_ALLOCATION_SIZE,
      const std::size_t growthFact = GROWTH_FACTOR)
      : initialAllocationSize(initialAllocSize), growthFactor(growthFact) {}

  /// default destructor
  ~ComplexCache() = default;

  /// Get the number of entries in the cache
  [[nodiscard]] std::size_t getCount() const { return count; }
  /// Get the peak number of entries in the cache
  [[nodiscard]] std::size_t getPeakCount() const { return peakCount; }
  /// Get the number of allocations performed
  [[nodiscard]] std::size_t getAllocations() const { return allocations; }
  /// Get the growth factor used for the allocation size
  [[nodiscard]] std::size_t getGrowthFactor() const { return growthFactor; }

  /**
   * @brief Get a Complex from the cache
   * @details This method returns a fresh Complex from the cache. It first
   * checks if there is an entry on the available list. If not, it checks
   * whether the current chunk has enough space left. If not, it allocates a new
   * chunk. It then returns a Complex from the current chunk. This consumes two
   * entries from the cache.
   * @return a Complex from the cache
   */
  [[nodiscard]] Complex getCachedComplex();

  /**
   * @brief Get a temporary Complex from the cache
   * @details In contrast to getCachedComplex(), this method does not consume
   * two entries from the cache. It just provides temporary access to a Complex
   * from the cache. Any subsequent call to getCachedComplex() or this method
   * will potentially invalidate the Complex returned by this method.
   * @see getCachedComplex()
   * @return
   */
  [[nodiscard]] Complex getTemporaryComplex();

  /**
   * @brief Return a Complex to the available list
   * @details This method returns a Complex to the available list. After this
   * call, the Complex should not be used anymore. The Complex must not be
   * Complex::zero or Complex::one. The Complex must not be in use, i.e. its
   * reference count must be zero.
   * @param c the Complex to return
   */
  void returnToCache(Complex& c);

  /**
   * @brief Clear the cache
   * @details This method clears the cache. It discards the available list and
   * all but the first chunk of memory. Also resets all counters.
   */
  void clear();

private:
  /// the list of entries that are available for reuse
  CTEntry* available{};
  /// the number of entries initially allocated
  std::size_t initialAllocationSize;
  /// the growth factor for the allocation size
  std::size_t growthFactor;
  /// the actual chunks of memory
  std::vector<std::vector<CTEntry>> chunks{
      1U, std::vector<CTEntry>(initialAllocationSize)};
  /// the current chunk
  std::size_t chunkID{};
  /// the iterator for the current chunk
  std::vector<CTEntry>::iterator chunkIt{chunks.front().begin()};
  /// the end iterator for the current chunk
  std::vector<CTEntry>::iterator chunkEndIt{chunks.front().end()};
  /// the size of the next allocation
  std::size_t allocationSize{initialAllocationSize * growthFactor};
  /// the total number of allocations performed
  std::size_t allocations = initialAllocationSize;
  /// the number of entries in the cache
  std::size_t count = 0;
  /// the peak number of entries in the cache
  std::size_t peakCount = 0;
};
} // namespace dd
