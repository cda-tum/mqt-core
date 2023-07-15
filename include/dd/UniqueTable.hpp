#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/UniqueTableStatistics.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace dd {

/**
 * @brief Data structure for uniquely storing DD nodes
 * @tparam Node class of nodes to provide/store
 * @tparam NBUCKET number of hash buckets to use (has to be a power of two)
 */
template <class Node, std::size_t NBUCKET = 32768> class UniqueTable {

public:
  /**
   * @brief The initial garbage collection limit.
   * @details The initial garbage collection limit is the number of entries that
   * must be present in the table before garbage collection is triggered.
   * Increasing this number reduces the number of garbage collections, but
   * increases the memory usage.
   */
  static constexpr std::size_t INITIAL_GC_LIMIT = 131072U;

  /**
   * @brief The default constructor
   * @param nv The number of variables
   * @param manager The memory manager to use for allocating new nodes.
   * @param initialGCLim The initial garbage collection limit.
   */
  explicit UniqueTable(const std::size_t nv, MemoryManager<Node>& manager,
                       std::size_t initialGCLim = INITIAL_GC_LIMIT)
      : nvars(nv), memoryManager(&manager), initialGCLimit(initialGCLim) {}

  void resize(std::size_t nq) {
    nvars = nq;
    tables.resize(nq);
    // TODO: if the new size is smaller than the old one we might have to
    // release the unique table entries for the superfluous variables
    active.resize(nq);
    stats.activeEntryCount = 0;
    for (auto i = 0U; i < nq; ++i) {
      stats.activeEntryCount += active[i];
    }
  }

  /**
   * @brief The hash function for the hash table.
   * @details The hash function just combines the hashes of the edges of the
   * node. The hash value is masked to ensure that it is in the range
   * [0, NBUCKET - 1].
   * @param p The node to hash.
   * @returns The hash value of the node.
   */
  static std::size_t hash(const Node* p) {
    static constexpr std::size_t MASK = NBUCKET - 1;
    std::size_t key = 0;
    for (std::size_t i = 0; i < p->e.size(); ++i) {
      key = combineHash(key, std::hash<Edge<Node>>{}(p->e[i]));
    }
    key &= MASK;
    return key;
  }

  /// Get a reference to the table
  [[nodiscard]] const auto& getTables() const { return tables; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  static bool nodesAreEqual(const Node* p, const Node* q) {
    if constexpr (std::is_same_v<Node, dNode>) {
      return (p->e == q->e && (p->flags == q->flags));
    } else {
      return p->e == q->e;
    }
  }

  // lookup a node in the unique table for the appropriate variable; insert it,
  // if it has not been found NOTE: reference counting is to be adjusted by
  // function invoking the table lookup and only normalized nodes shall be
  // stored.
  Edge<Node> lookup(const Edge<Node>& e, bool keepNode = false) {
    // there are unique terminal nodes
    if (e.isTerminal()) {
      return e;
    }

    ++stats.lookups;
    const auto key = hash(e.p);
    const auto v = e.p->v;

    // search bucket in table corresponding to hashed value for the given node
    // and return it if found.
    if (const auto hashedNode = searchTable(e, key, keepNode);
        !hashedNode.isZeroTerminal()) {
      return hashedNode;
    }

    // if node not found -> add it to front of unique table bucket
    e.p->next = tables[v][key];
    tables[v][key] = e.p;
    stats.trackInsert();

    return e;
  }

  /**
   * @brief Increment the reference count of a node.
   * @details This is a pass-through function that calls the increment function
   * of the node. It additionally keeps track of the number of active entries
   * in the table (entries with a reference count greater than zero). Reference
   * counts saturate at the maximum value of RefCount.
   * @param p A pointer to the node to increase the reference count of.
   * @returns Whether the reference count was increased.
   * @see Node::incRef(Node*)
   */
  [[nodiscard]] bool incRef(Node* p) noexcept {
    const auto inc = ::dd::incRef(p);
    if (inc && p->ref == 1U) {
      stats.trackActiveEntry();
      ++active[p->v];
    }
    return inc;
  }

  /**
   * @brief Decrement the reference count of a node.
   * @details This is a pass-through function that calls the decrement function
   * of the node. It additionally keeps track of the number of active entries
   * in the table (entries with a reference count greater than zero). Reference
   * counts saturate at the maximum value of RefCount.
   * @param p A pointer to the node to decrease the reference count of.
   * @returns Whether the reference count was decreased.
   * @see Node::decRef(Node*)
   */
  [[nodiscard]] bool decRef(Node* p) noexcept {
    const auto dec = ::dd::decRef(p);
    if (dec && p->ref == 0U) {
      --stats.activeEntryCount;
      --active[p->v];
    }
    return dec;
  }

  [[nodiscard]] bool possiblyNeedsCollection() const {
    return stats.entryCount >= gcLimit;
  }

  std::size_t garbageCollect(bool force = false) {
    ++stats.gcCalls;
    if ((!force && !possiblyNeedsCollection()) || stats.entryCount == 0) {
      return 0;
    }

    ++stats.gcRuns;
    const auto entryCountBefore = stats.entryCount;
    for (auto& table : tables) {
      for (auto& bucket : table) {
        Node* p = bucket;
        Node* lastp = nullptr;
        while (p != nullptr) {
          if (p->ref == 0) {
            Node* next = p->next;
            if (lastp == nullptr) {
              bucket = next;
            } else {
              lastp->next = next;
            }
            memoryManager->returnEntry(p);
            p = next;
            --stats.entryCount;
          } else {
            lastp = p;
            p = p->next;
          }
        }
      }
    }
    // The garbage collection limit changes dynamically depending on the number
    // of remaining (active) nodes. If it were not changed, garbage collection
    // would run through the complete table on each successive call once the
    // number of remaining entries reaches the garbage collection limit. It is
    // increased whenever the number of remaining entries is rather close to the
    // garbage collection threshold and decreased if the number of remaining
    // entries is much lower than the current limit.
    if (stats.entryCount > gcLimit / 10 * 9) {
      gcLimit = stats.entryCount + initialGCLimit;
    }
    return entryCountBefore - stats.entryCount;
  }

  void clear() {
    // clear unique table buckets
    for (auto& table : tables) {
      for (auto& bucket : table) {
        bucket = nullptr;
      }
    }
    gcLimit = initialGCLimit;
    std::fill(active.begin(), active.end(), 0);
    stats.reset();
  };

  void print() {
    auto q = nvars - 1U;
    for (auto it = tables.rbegin(); it != tables.rend(); ++it) {
      auto& table = *it;
      std::cout << "\tq" << q << ":"
                << "\n";
      for (std::size_t key = 0; key < table.size(); ++key) {
        auto p = table[key];
        if (p != nullptr) {
          std::cout << "\tkey=" << key << ": ";
        }

        while (p != nullptr) {
          std::cout << "\t\t" << std::hex << reinterpret_cast<std::uintptr_t>(p)
                    << std::dec << " " << p->ref << std::hex;
          for (const auto& e : p->e) {
            std::cout << " p" << reinterpret_cast<std::uintptr_t>(e.p) << "(r"
                      << reinterpret_cast<std::uintptr_t>(e.w.r) << " i"
                      << reinterpret_cast<std::uintptr_t>(e.w.i) << ")";
          }
          std::cout << std::dec << "\n";
          p = p->next;
        }
      }
      --q;
    }
  }

private:
  /// Typedef for a bucket in the table
  using Bucket = Node*;
  /// Typedef for the table
  using Table = std::array<Bucket, NBUCKET>;

  /// The number of variables
  std::size_t nvars = 0U;
  /**
   * @brief The actual tables (one for each variable)
   * @details Each hash table is an array of buckets. Each bucket is a linked
   * list of entries. The linked list is implemented by using the next pointer
   * of the entries.
   */
  std::vector<Table> tables{nvars};

  /// A pointer to the memory manager for the nodes stored in the table.
  MemoryManager<Node>* memoryManager;

  /// A collection of statistics
  UniqueTableStatistics stats{};

  /**
   * @brief the number of active nodes for each variable
   * @note A node is considered active if it has a non-zero reference count.
   */
  std::vector<std::size_t> active{std::vector<std::size_t>(nvars, 0U)};

  /// The initial garbage collection limit
  std::size_t initialGCLimit;
  /// The current garbage collection limit
  std::size_t gcLimit = initialGCLimit;

  /**
  Searches for a node in the hash table with the given key.
  @param e The node to search for.
  @param key The hashed value used to search the table.
  @param keepNode If true, the node pointed to by e.p will not be put on the
  available chain.
  @return The Edge<Node> found in the hash table or Edge<Node>::zero if not
  found.
  **/
  Edge<Node> searchTable(const Edge<Node>& e, const std::size_t& key,
                         const bool keepNode = false) {
    Node* p = tables[e.p->v][key];
    while (p != nullptr) {
      if (nodesAreEqual(e.p, p)) {
        // Match found
        if (e.p != p && !keepNode) {
          // put node pointed to by e.p on available chain
          memoryManager->returnEntry(e.p);
        }
        ++stats.hits;

        // variables should stay the same
        assert(p->v == e.p->v);

        return {p, e.w};
      }
      ++stats.collisions;
      p = p->next;
    }

    // Node not found in bucket
    return Edge<Node>::zero;
  }
};

} // namespace dd
