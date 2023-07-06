#pragma once

#include "dd/ComplexNumbers.hpp"
#include "dd/Definitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace dd {

/**
 * @brief Data structure for providing and uniquely storing DD nodes
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
    activeNodeCount = std::accumulate(active.begin(), active.end(),
                                      static_cast<std::size_t>(0U));
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
      key = dd::combineHash(key, std::hash<Edge<Node>>{}(p->e[i]));
    }
    key &= MASK;
    return key;
  }

  /// Get a reference to the table
  [[nodiscard]] const auto& getTables() const { return tables; }
  /// Get the current node count
  [[nodiscard]] std::size_t getNodeCount() const { return nodeCount; }
  /// Get the peak node count
  [[nodiscard]] std::size_t getPeakNodeCount() const { return peakNodeCount; }
  /// Get the maximum number of active nodes
  [[nodiscard]] std::size_t getMaxActiveNodes() const { return maxActive; }

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

    ++lookups;
    const auto key = hash(e.p);
    const auto v = e.p->v;

    // search bucket in table corresponding to hashed value for the given node
    // and return it if found.
    if (const auto hashedNode = searchTable(e, key, keepNode);
        !hashedNode.isZeroTerminal()) {
      return hashedNode;
    }

    // if node not found -> add it to front of unique table bucket
    e.p->next = tables[static_cast<std::size_t>(v)][key];
    tables[static_cast<std::size_t>(v)][key] = e.p;
    ++inserts;
    ++nodeCount;
    peakNodeCount = std::max(peakNodeCount, nodeCount);

    return e;
  }

  // increment reference counter for node e points to
  // and recursively increment reference counter for
  // each child if this is the first reference
  void incRef(const Edge<Node>& e) {
    dd::ComplexNumbers::incRef(e.w);
    if (e.p == nullptr || e.isTerminal()) {
      return;
    }

    if (e.p->ref == std::numeric_limits<decltype(e.p->ref)>::max()) {
      std::clog << "[WARN] MAXREFCNT reached for p="
                << reinterpret_cast<std::uintptr_t>(e.p)
                << ". Node will never be collected.\n";
      return;
    }

    e.p->ref++;

    if (e.p->ref == 1) {
      for (const auto& edge : e.p->e) {
        if (edge.p != nullptr) {
          incRef(edge);
        }
      }
      active[static_cast<std::size_t>(e.p->v)]++;
      activeNodeCount++;
      maxActive = std::max(maxActive, activeNodeCount);
    }
  }

  // decrement reference counter for node e points to
  // and recursively decrement reference counter for
  // each child if this is the last reference
  void decRef(const Edge<Node>& e) {
    dd::ComplexNumbers::decRef(e.w);
    if (e.p == nullptr || e.isTerminal()) {
      return;
    }
    if (e.p->ref == std::numeric_limits<decltype(e.p->ref)>::max()) {
      return;
    }

    if (e.p->ref == 0) {
      throw std::runtime_error("In decref: ref==0 before decref\n");
    }

    e.p->ref--;

    if (e.p->ref == 0) {
      for (const auto& edge : e.p->e) {
        if (edge.p != nullptr) {
          decRef(edge);
        }
      }
      active[static_cast<std::size_t>(e.p->v)]--;
      activeNodeCount--;
    }
  }

  [[nodiscard]] bool possiblyNeedsCollection() const {
    return nodeCount >= gcLimit;
  }

  std::size_t garbageCollect(bool force = false) {
    gcCalls++;
    if ((!force && nodeCount < gcLimit) || nodeCount == 0) {
      return 0;
    }

    gcRuns++;
    std::size_t collected = 0;
    std::size_t remaining = 0;
    for (auto& table : tables) {
      for (auto& bucket : table) {
        Node* p = bucket;
        Node* lastp = nullptr;
        while (p != nullptr) {
          if (p->ref == 0) {
            assert(!Node::isTerminal(p));
            Node* next = p->next;
            if (lastp == nullptr) {
              bucket = next;
            } else {
              lastp->next = next;
            }
            memoryManager->returnEntry(p);
            p = next;
            collected++;
          } else {
            lastp = p;
            p = p->next;
            remaining++;
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
    if (remaining > gcLimit / 10 * 9) {
      gcLimit = remaining + initialGCLimit;
    }
    nodeCount = remaining;
    return collected;
  }

  void clear() {
    // clear unique table buckets
    for (auto& table : tables) {
      for (auto& bucket : table) {
        bucket = nullptr;
      }
    }

    nodeCount = 0;
    peakNodeCount = 0;

    collisions = 0;
    hits = 0;
    lookups = 0;
    inserts = 0;

    std::fill(active.begin(), active.end(), 0);
    activeNodeCount = 0;
    maxActive = 0;

    gcCalls = 0;
    gcRuns = 0;
    gcLimit = initialGCLimit;
  };

  void print() {
    auto q = static_cast<dd::Qubit>(nvars - 1);
    for (auto it = tables.rbegin(); it != tables.rend(); ++it) {
      auto& table = *it;
      std::cout << "\tq" << static_cast<std::size_t>(q) << ":"
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

  void printActive() {
    std::cout << "#printActive: " << activeNodeCount << ", ";
    for (const auto& a : active) {
      std::cout << a << " ";
    }
    std::cout << "\n";
  }

  /**
   * @brief Get the hit ratio of the table.
   * @details The hit ratio is the ratio of lookups that were successful.
   * @returns The hit ratio of the table.
   */
  [[nodiscard]] fp hitRatio() const noexcept {
    if (lookups == 0) {
      return 0.;
    }
    return static_cast<fp>(hits) / static_cast<fp>(lookups);
  }

  /**
   * @brief Get the collision ratio of the table.
   * @details A collision occurs when the hash function maps two different
   * floating point numbers to the same bucket. The collision ratio is the ratio
   * of lookups that resulted in a collision.
   * @returns The collision ratio of the table.
   */
  [[nodiscard]] fp colRatio() const noexcept {
    if (lookups == 0) {
      return 0.;
    }
    return static_cast<fp>(collisions) / static_cast<fp>(lookups);
  }

  [[nodiscard]] std::size_t getActiveNodeCount() const noexcept {
    return activeNodeCount;
  }

  [[nodiscard]] std::size_t getActiveNodeCount(Qubit var) const {
    return active.at(var);
  }

  std::ostream& printStatistics(std::ostream& os = std::cout) const {
    os << "hits: " << hits << ", collisions: " << collisions
       << ", looks: " << lookups << ", inserts: " << inserts
       << ", hitRatio: " << hitRatio() << ", colRatio: " << colRatio()
       << ", gc calls: " << gcCalls << ", gc runs: " << gcRuns << "\n";
    return os;
  }

private:
  /// Typedef for a bucket in the table
  using Bucket = Node*;
  /// Typedef for the table
  using Table = std::array<Bucket, NBUCKET>;

  /// The number of variables
  std::size_t nvars = 0;
  /**
   * @brief The actual tables (one for each variable)
   * @details Each hash table is an array of buckets. Each bucket is a linked
   * list of entries. The linked list is implemented by using the next pointer
   * of the entries.
   */
  std::vector<Table> tables{nvars};

  /// A pointer to the memory manager for the numbers stored in the table.
  MemoryManager<Node>* memoryManager{};

  /// The number of collisions
  std::size_t collisions = 0;
  /// The number of successful lookups
  std::size_t hits = 0;
  /// The number of lookups
  std::size_t lookups = 0;
  /// The number of inserts
  std::size_t inserts = 0;

  /// The number of nodes in the table
  std::size_t nodeCount = 0U;
  /// The peak number of nodes in the table
  std::size_t peakNodeCount = 0U;
  /**
   * @brief the number of active nodes for each variable
   * @note A node is considered active if it has a non-zero reference count.
   */
  std::vector<std::size_t> active{std::vector<std::size_t>(nvars, 0)};
  /// The total number of active nodes
  std::size_t activeNodeCount = 0;
  /// The maximum number of active nodes
  std::size_t maxActive = 0;

  /// The initial garbage collection limit
  std::size_t initialGCLimit;
  /// The number of garbage collection calls
  std::size_t gcCalls = 0;
  /// The number of garbage actual garbage collection runs
  std::size_t gcRuns = 0;
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
    const auto v = e.p->v;

    Node* p = tables[static_cast<std::size_t>(v)][key];
    while (p != nullptr) {
      if (nodesAreEqual(e.p, p)) {
        // Match found
        if (e.p != p && !keepNode) {
          // put node pointed to by e.p on available chain
          memoryManager->returnEntry(e.p);
        }
        ++hits;

        // variables should stay the same
        assert(p->v == e.p->v);

        return {p, e.w};
      }
      ++collisions;
      p = p->next;
    }

    // Node not found in bucket
    return Edge<Node>::zero;
  }
};

} // namespace dd
