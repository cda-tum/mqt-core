#pragma once

#include "dd/Definitions.hpp"
#include "dd/MemoryManager.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace dd {

struct RealNumber;

/**
 * @brief A unique table for real numbers.
 * @details A hash table that stores real numbers. The hash table is implemented
 * as an array of buckets, each of which is a linked list of entries. The hash
 * table has a fixed number of buckets.
 * @note: The implementation assumes that all values are non-negative and in the
 * range [0, 1]. While numbers outside of this range can be stored, they will
 * always be placed in the same bucket and will therefore cause collisions.
 */
class RealNumberUniqueTable {
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
   * @brief The initial garbage collection limit.
   * @details The initial garbage collection limit is the number of entries that
   * must be present in the table before garbage collection is triggered.
   * Increasing this number reduces the number of garbage collections, but
   * increases the memory usage.
   */
  static constexpr std::size_t INITIAL_GC_LIMIT = 65536U;

public:
  /// The default constructor.
  explicit RealNumberUniqueTable(MemoryManager<RealNumber>& manager,
                                 std::size_t initialGCLim = INITIAL_GC_LIMIT);

  /// Get the numerical tolerance used for floating point comparisons.
  static fp tolerance() noexcept;

  /// Set the numerical tolerance used for floating point comparisons.
  static void setTolerance(fp tol) noexcept;

  /**
   * @brief The hash function for the hash table.
   * @details The hash function for the table is a simple linear (clipped) hash
   * function. The hash function is used to map floating point numbers to the
   * buckets of the table.
   * @param val The floating point number to hash. Must be non-negative.
   * @returns The hash value of the floating point number.
   */
  static std::int64_t hash(fp val) noexcept;

  /// Get a reference to the table.
  [[nodiscard]] const auto& getTable() const noexcept { return table; }

  /**
   * @brief Lookup a number in the table.
   * @details This function looks up a number in the table. If the number is not
   * found, a new number is created and inserted into the table.
   * @param val The floating point number to look up. Must be non-negative.
   * @returns A pointer to the number corresponding to the input number.
   */
  [[nodiscard]] RealNumber* lookup(fp val);

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
  void print() const;

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
  using Bucket = RealNumber*;
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
  std::array<RealNumber*, NBUCKET> tailTable{};

  /// A pointer to the memory manager for the numbers stored in the table.
  MemoryManager<RealNumber>* memoryManager{};

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
  RealNumber* findOrInsert(std::int64_t key, fp val);

  /**
   * @brief Inserts a value into the bucket indexed by key.
   * @details This function inserts a value into the bucket indexed by key.
   * It assumes that no element within TOLERANCE is present in the bucket.
   * @param key The index of the bucket to insert the value into.
   * @param val The value to insert.
   * @returns A pointer to the inserted entry.
   */
  RealNumber* insert(std::int64_t key, fp val);
};
} // namespace dd
