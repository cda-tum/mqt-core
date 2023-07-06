#pragma once

#include "Definitions.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace dd {
/**
 * @brief A hash table for complex numbers.
 * @details The complex table is a hash table that stores complex numbers as
 * pairs of floating point numbers. The hash table is implemented as an array
 * of buckets, each of which is a linked list of entries. The hash table has a
 * fixed number of buckets.
 * @note This class is historically misnamed. It does not store complex numbers,
 * but rather floating point numbers. It is used as the basis for storing
 * complex numbers in the DD package. The name is kept for historical reasons.
 */
class ComplexTable {
  /**
   * @brief The number of buckets in the table.
   * @details The number of buckets is fixed and cannot be changed after the
   * table has been created. Increasing the number of buckets reduces the
   * number of collisions, but increases the memory usage.
   * @attention The number of buckets has to be one larger than a power of two.
   * Otherwise, the hash function will not work correctly.
   */
  static constexpr std::size_t NBUCKET = 65537U;

  /**
   * @brief The number of initially allocated entries.
   * @details The number of initially allocated entries is the number of entries
   * that are allocated as a chunk when the table is created. Increasing this
   * number reduces the number of allocations, but increases the memory usage.
   */
  static constexpr std::size_t INITIAL_ALLOCATION_SIZE = 2048U;
  /**
   * @brief The growth factor for table entry allocation.
   * @details The growth factor is used to determine the number of entries that
   * are allocated when the table runs out of entries. Per default, the number
   * of entries is doubled. Increasing this number reduces the number of memory
   * allocations, but increases the memory usage.
   */
  static constexpr std::size_t GROWTH_FACTOR = 2U;
  /**
   * @brief The initial garbage collection limit.
   * @details The initial garbage collection limit is the number of entries that
   * must be present in the table before garbage collection is triggered.
   * Increasing this number reduces the number of garbage collections, but
   * increases the memory usage.
   */
  static constexpr std::size_t INITIAL_GC_LIMIT = 65536U;

public:
  /**
   * @brief An entry in the table.
   * @details An entry in the table consists of a floating point number (the
   * value), a next pointer, and a reference count.
   *
   * @note Due to the way the sign of the value is encoded, special care has to
   * be taken when accessing the value. The static functions in this struct
   * provide safe access to the value.
   */
  struct Entry {
    /**
     * @brief The value of the entry.
     * @details The value of the entry is a floating point number. The sign of
     * the value is encoded in the least significant bit of the memory address
     * of the entry. As a consequence, values stored in the table are always
     * non-negative. The sign of the value as well as the value itself can be
     * accessed using the static functions of this struct.
     */
    fp value{};
    /**
     * @brief The next pointer of the entry.
     * @details The next pointer is used to chain entries in the same bucket.
     * The next pointer is part of the entry struct for efficiency reasons. It
     * could be stored separately, but that would require many small
     * allocations. This way, the entries can be allocated in chunks, which is
     * much more efficient.
     */
    Entry* next{};
    /**
     * @brief The reference count of the entry.
     * @details The reference count is used to determine whether an entry is
     * still in use. If the reference count is zero, the entry is not in use and
     * can be garbage collected.
     */
    RefCount refCount{};

    /**
     * @brief Get an aligned pointer to the entry.
     * @details Since the least significant bit of the memory address of the
     * entry is used to encode the sign of the value, the pointer to the entry
     * might not be aligned. This function returns an aligned pointer to the
     * entry.
     * @param e The entry to get the aligned pointer for.
     * @returns An aligned pointer to the entry.
     */
    [[nodiscard]] static Entry* getAlignedPointer(const Entry* e) noexcept;

    /**
     * @brief Get a pointer to the entry with a negative sign.
     * @details Since the least significant bit of the memory address of the
     * entry is used to encode the sign of the value, this function just sets
     * the least significant bit of the memory address of the entry to 1.
     * @param e The entry to get the negative pointer for.
     * @returns A negative pointer to the entry.
     */
    [[nodiscard]] static Entry* getNegativePointer(const Entry* e) noexcept;

    /**
     * @brief Check whether the entry is a negative pointer.
     * @param e The entry to check.
     * @returns Whether the entry is a negative pointer.
     */
    [[nodiscard]] static bool isNegativePointer(const Entry* e) noexcept;

    /**
     * @brief Flip the sign of the entry pointer.
     * @param e The entry to flip the sign of.
     * @returns The entry with the sign flipped.
     * @note This function does not change the sign of the value of the entry.
     * It rather changes the sign of the pointer to the entry.
     * @note We do not consider negative zero here, since it is not used in the
     * DD package. There only exists one zero entry, which is positive.
     */
    [[nodiscard]] static Entry* flipPointerSign(const Entry* e) noexcept;

    /**
     * @brief Check whether the entry points to the zero entry.
     * @param e The entry to check.
     * @returns Whether the entry points to zero.
     */
    [[nodiscard]] static bool exactlyZero(const Entry* e) noexcept;

    /**
     * @brief Check whether the entry points to the one entry.
     * @param e The entry to check.
     * @returns Whether the entry points to one.
     */
    [[nodiscard]] static bool exactlyOne(const Entry* e) noexcept;

    /**
     * @brief Get the value of the entry.
     * @param e The entry to get the value for.
     * @returns The value of the entry.
     * @note This function accounts for the sign of the entry embedded in the
     * memory address of the entry.
     */
    [[nodiscard]] static fp val(const Entry* e) noexcept;

    /**
     * @brief Get the reference count of the entry.
     * @param e The entry to get the reference count for.
     * @returns The reference count of the entry.
     * @note This function accounts for the sign of the entry embedded in the
     * memory address of the entry.
     */
    [[nodiscard]] static RefCount ref(const Entry* e) noexcept;

    /**
     * @brief Check whether two floating point numbers are approximately equal.
     * @details This function checks whether two floating point numbers are
     * approximately equal. The two numbers are considered approximately equal
     * if the absolute difference between them is smaller than a small value
     * (TOLERANCE). This function is used to compare floating point numbers
     * stored in the table.
     * @param left The first floating point number.
     * @param right The second floating point number.
     * @returns Whether the two floating point numbers are approximately equal.
     */
    [[nodiscard]] static bool approximatelyEquals(fp left, fp right) noexcept;

    /**
     * @brief Check whether two entries are approximately equal.
     * @details This function checks whether two entries are approximately
     * equal. Two entries are considered approximately equal if they point to
     * the same entry or if the values of the entries are approximately equal.
     * @param left The first entry.
     * @param right The second entry.
     * @returns Whether the two entries are approximately equal.
     * @see approximatelyEquals(fp, fp)
     */
    [[nodiscard]] static bool approximatelyEquals(const Entry* left,
                                                  const Entry* right) noexcept;

    /**
     * @brief Check whether a floating point number is approximately zero.
     * @param e The floating point number to check.
     * @returns Whether the floating point number is approximately zero.
     */
    [[nodiscard]] static bool approximatelyZero(fp e) noexcept;

    /**
     * @brief Check whether an entry is approximately zero.
     * @param e The entry to check.
     * @returns Whether the entry is approximately zero.
     * @see approximatelyZero(fp)
     */
    [[nodiscard]] static bool approximatelyZero(const Entry* e) noexcept;

    /**
     * @brief Check whether a floating point number is approximately one.
     * @param e The floating point number to check.
     * @returns Whether the floating point number is approximately one.
     */
    [[nodiscard]] static bool approximatelyOne(fp e) noexcept;

    /**
     * @brief Check whether an entry is approximately one.
     * @param e The entry to check.
     * @returns Whether the entry is approximately one.
     * @see approximatelyOne(fp)
     */
    [[nodiscard]] static bool approximatelyOne(const Entry* e) noexcept;

    /**
     * @brief Write a binary representation of the entry to a stream.
     * @param e The entry to write.
     * @param os The stream to write to.
     */
    static void writeBinary(const Entry* e, std::ostream& os);
  };

  // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
  /// The constant zero entry.
  static inline Entry zero{0., nullptr, 1U};
  /// The constant one entry.
  static inline Entry one{1., nullptr, 1U};
  /// The constant sqrt(2)/2 = 1/sqrt(2) entry.
  static inline Entry sqrt2over2{SQRT2_2, nullptr, 1U};
  // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

  /**
   * @brief The default constructor.
   * @details The default constructor initializes the complex table with the
   * default values for the initial allocation size, the growth factor, and the
   * initial garbage collection limit.
   * @param initialAllocSize The initial allocation size.
   * @param growthFact The growth factor used for the allocation size.
   * @param initialGCLim The initial garbage collection limit.
   *
   * @see INITIAL_ALLOCATION_SIZE
   * @see GROWTH_FACTOR
   * @see INITIAL_GC_LIMIT
   */
  explicit ComplexTable(std::size_t initialAllocSize = INITIAL_ALLOCATION_SIZE,
                        std::size_t growthFact = GROWTH_FACTOR,
                        std::size_t initialGCLim = INITIAL_GC_LIMIT);

  /// The default destructor.
  ~ComplexTable() = default;

  /**
   * @brief Getter for the tolerance used for floating point comparisons.
   * @returns The tolerance used for floating point comparisons.
   */
  static fp tolerance() noexcept { return TOLERANCE; }

  /**
   * @brief Setter for the tolerance used for floating point comparisons.
   * @param tol The tolerance used for floating point comparisons.
   */
  static void setTolerance(const fp tol) noexcept { TOLERANCE = tol; }

  /// The bit mask used for the hash function.
  static constexpr std::int64_t MASK = NBUCKET - 1;

  /**
   * @brief The hash function for the complex table.
   * @details The hash function for the complex table is a simple linear
   * (clipped) hash function. The hash function is used to map floating point
   * numbers to the buckets of the table.
   * @param val The floating point number to hash. Must be non-negative.
   * @returns The hash value of the floating point number.
   */
  static std::int64_t hash(const fp val) {
    assert(val >= 0);
    const auto key = static_cast<std::int64_t>(std::nearbyint(val * MASK));
    return std::min<std::int64_t>(key, MASK);
  }

  /// Get the number of entries in the table.
  [[nodiscard]] std::size_t getCount() const noexcept { return count; }

  /// Get the peak number of entries in the table.
  [[nodiscard]] std::size_t getPeakCount() const noexcept { return peakCount; }

  /// Get the number of allocations performed.
  [[nodiscard]] std::size_t getAllocations() const noexcept {
    return allocations;
  }

  /// Get the growth factor used for the allocation size.
  [[nodiscard]] std::size_t getGrowthFactor() const noexcept {
    return growthFactor;
  }

  /// Get a reference to the table.
  [[nodiscard]] const auto& getTable() const noexcept { return table; }

  /// Check whether there is an entry on the available list.
  [[nodiscard]] bool availableEmpty() const noexcept {
    return available == nullptr;
  }

  /**
   * @brief Lookup an entry in the table.
   * @details This function looks up an entry in the table. If the entry is not
   * found, a new entry is created and inserted into the table.
   * @param val The floating point number to look up. Must be non-negative.
   * @returns A pointer to the entry corresponding to the floating point number.
   */
  [[nodiscard]] Entry* lookup(fp val);

  /**
   * @brief Get a free entry
   * @details This function returns a free entry. It first checks whether there
   * is an entry on the available list. If not, it checks whether the currently
   * allocated chunk still has free entries. If not, it allocates a new chunk.
   * @returns A pointer to a free entry.
   */
  [[nodiscard]] Entry* getEntry();

  /**
   * @brief Return an entry to the available list.
   * @details This function marks an entry as available by appending it to the
   * available list. The entry is not actually deleted, but can be reused by
   * subsequent calls to getEntry().
   * @param entry
   */
  void returnEntry(Entry* entry) noexcept;

  /**
   * @brief Check whether an entry is one of the static entries.
   * @param entry The entry to check.
   * @returns Whether the entry is one of the static entries.
   */
  [[nodiscard]] static bool isStaticEntry(const Entry* entry) noexcept {
    return entry == &zero || entry == &one || entry == &sqrt2over2;
  }

  /**
   * @brief Increment the reference count for an entry.
   * @param entry The entry to increment the reference count for.
   */
  static void incRef(Entry* entry) noexcept;

  /**
   * @brief Decrement the reference count for an entry.
   * @param entry The entry to decrement the reference count for.
   */
  static void decRef(Entry* entry) noexcept;

  /**
   * @brief Check whether the table possibly needs garbage collection.
   * @returns Whether the number of entries in the table has reached the garbage
   * collection limit.
   */
  [[nodiscard]] bool possiblyNeedsCollection() const noexcept;

  /**
   * @brief Perform garbage collection.
   * @details This function performs garbage collection. It first checks whether
   * garbage collection is necessary. If not, it does nothing. Otherwise, it
   * iterates over all entries in the table and returns all entries with a
   * reference count of zero to the available list. If the force flag is set,
   * garbage collection is performed even if it is not necessary.
   * Based on how many entries are returned to the available list, the garbage
   * collection limit is dynamically adjusted.
   * @param force Whether to force garbage collection.
   * @returns The number of entries returned to the available list.
   */
  std::size_t garbageCollect(bool force = false) noexcept;

  /**
   * @brief Clear the table.
   * @details This function clears the table. It iterates over all entries in
   * the table and sets them to nullptr. It also discards the available list and
   * all but the first chunk of the allocated chunks. Also resets all counters.
   */
  void clear() noexcept;

  /**
   * @brief Print the table.
   */
  void print();

  /**
   * @brief Get the hit ratio of the table.
   * @details The hit ratio is the ratio of lookups that were successful.
   * @returns The hit ratio of the table.
   */
  [[nodiscard]] fp hitRatio() const noexcept;

  /**
   * @brief Get the collision ratio of the table.
   * @details A collision occurs when the hash function maps two different
   * floating point numbers to the same bucket. The collision ratio is the ratio
   * of lookups that resulted in a collision.
   * @returns The collision ratio of the table.
   */
  [[nodiscard]] fp colRatio() const noexcept;

  /**
   * @brief Get the statistics of the table.
   * @details The statistics of the table are the number of hits, collisions,
   * lookups, inserts, insert collisions, findOrInserts, upper neighbors, lower
   * neighbors, garbage collection calls and garbage collection runs.
   * @returns A map containing the statistics of the table.
   */
  std::map<std::string, std::size_t, std::less<>> getStatistics() noexcept;

  /**
   * @brief Print the statistics of the table.
   * @param os The output stream to print to.
   * @returns The output stream.
   */
  std::ostream& printStatistics(std::ostream& os = std::cout) const;

  /**
   * @brief Print the bucket distribution of the table.
   * @param os The output stream to print to.
   * @returns The output stream.
   */
  std::ostream& printBucketDistribution(std::ostream& os = std::cout);

private:
  /// Typedef for a bucket in the table.
  using Bucket = Entry*;
  /// Typedef for the table.
  using Table = std::array<Bucket, NBUCKET>;

  /**
   * @brief The actual hash table
   * @details The hash table is an array of buckets. Each bucket is a linked
   * list of entries. The linked list is implemented by using the next pointer
   * of the entries.
   */
  Table table{};
  /**
   * @brief The tail table
   * @details The tail table is an array of pointers to the last entry in each
   * bucket. This is used to speed up the insertion of new entries.
   */
  std::array<Entry*, NBUCKET> tailTable{};

  /// the number of collisions
  std::size_t collisions = 0;
  /// the number of collisions when inserting
  std::size_t insertCollisions = 0;
  /// the number of successful lookups
  std::size_t hits = 0;
  /// the number of calls to findOrInsert
  std::size_t findOrInserts = 0;
  /// the number of lookups
  std::size_t lookups = 0;
  /// the number of inserts
  std::size_t inserts = 0;
  /// the number of borderline cases where the lower neighbor is returned
  std::size_t lowerNeighbors = 0;
  /// the number of borderline cases where the upper neighbor is returned
  std::size_t upperNeighbors = 0;

  /// numerical tolerance to be used for floating point values
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,readability-identifier-naming)
  static inline fp TOLERANCE = std::numeric_limits<dd::fp>::epsilon() * 1024;

  /// the list of entries that are available for reuse
  Entry* available{};
  /// the number of entries initially allocated
  std::size_t initialAllocationSize;
  /// the growth factor for the number of entries
  std::size_t growthFactor;
  /// the actual memory chunks of entries
  std::vector<std::vector<Entry>> chunks{
      1U, std::vector<Entry>{initialAllocationSize}};
  /// the current chunk
  std::size_t chunkID{};
  /// the iterator for the current chunk
  std::vector<Entry>::iterator chunkIt{chunks.front().begin()};
  /// the end iterator for the current chunk
  std::vector<Entry>::iterator chunkEndIt{chunks.front().end()};
  /// the size of the next allocation
  std::size_t allocationSize{initialAllocationSize * growthFactor};
  /// the total number of allocations performed
  std::size_t allocations = initialAllocationSize;
  /// the number of entries in the table
  std::size_t count = 0;
  /// the peak number of entries in the table
  std::size_t peakCount = 0;

  /// the initial garbage collection limit
  std::size_t initialGCLimit;
  /// the number of garbage collection calls
  std::size_t gcCalls = 0;
  /// the number of garbage actual garbage collection runs
  std::size_t gcRuns = 0;
  /// the current garbage collection limit
  std::size_t gcLimit = initialGCLimit;

  /**
   * @brief Finds or inserts a value into the bucket indexed by key.
   * @details This function either finds an entry with a value within TOLERANCE
   * of val in the bucket indexed by key or inserts a new entry with value val
   * into the bucket.
   * @param key The index of the bucket to find or insert the value into.
   * @param val The value to find or insert.
   * @returns A pointer to the found or inserted entry.
   */
  Entry* findOrInsert(std::int64_t key, fp val);

  /**
   * @brief Inserts a value into the bucket indexed by key.
   * @details This function inserts a value into the bucket indexed by key.
   * It assumes that no element within TOLERANCE is present in the bucket.
   * @param key The index of the bucket to insert the value into.
   * @param val The value to insert.
   * @returns A pointer to the inserted entry.
   */
  Entry* insert(std::int64_t key, fp val);
};
/// Alias for ComplexTable::Entry
using CTEntry = ComplexTable::Entry;
} // namespace dd
