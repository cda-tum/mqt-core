/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Definitions.hpp"
#include "dd/Edge.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/statistics/UniqueTableStatistics.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <type_traits>
#include <vector>

namespace dd {

/**
 * @brief Data structure for uniquely storing DD nodes
 */
class UniqueTable {
public:
  /**
   * @brief The initial garbage collection limit.
   * @details The initial garbage collection limit is the number of entries that
   * must be present in the table before garbage collection is triggered.
   * Increasing this number reduces the number of garbage collections, but
   * increases the memory usage.
   */
  static constexpr std::size_t INITIAL_GC_LIMIT = 131072U;

  struct UniqueTableConfig {
    /// The number of variables
    std::size_t nVars = 0U;

    /// number of hash buckets to use (has to be a power of two)
    const std::size_t nBuckets = 32768;

    /// The initial garbage collection limit
    const std::size_t initialGCLimit = INITIAL_GC_LIMIT;
  };

  /**
   * @brief The default constructor
   * @param manager The memory manager to use
   * @param config The configuration for the unique table
   * @details The MemoryManager shall be constructed from the same type that the
   * unique table is then used for in the lookup method.
   */
  UniqueTable(MemoryManager& manager, UniqueTableConfig config);

  void resize(std::size_t nVars);

  /**
   * @brief The hash function for the hash table.
   * @details The hash function just combines the hashes of the edges of the
   * node. The hash value is masked to ensure that it is in the range
   * [0, nBuckets - 1].
   * @param p The node to hash.
   * @returns The hash value of the node.
   */
  template <class Node> std::size_t hash(const Node* p) {
    const std::size_t MASK = cfg.nBuckets - 1;
    std::size_t key = 0U;
    for (std::size_t i = 0U; i < p->e.size(); ++i) {
      qc::hashCombine(key, std::hash<Edge<Node>>{}(p->e[i]));
    }
    key &= MASK;
    return key;
  }

  template <class Node>
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
  template <class Node> Node* lookup(Node* p) {
    // there are unique terminal nodes
    if (NodeBase::isTerminal(p)) {
      return p;
    }

    const auto key = hash(p);
    const auto v = p->v;
    ++stats[v].lookups;

    // search bucket in table corresponding to hashed value for the given node
    // and return it if found.
    if (auto* hashedNode = searchTable(p, key); !Node::isTerminal(hashedNode)) {
      return hashedNode;
    }

    // if node not found -> add it to front of unique table bucket
    p->setNext(tables[v][key]);
    tables[v][key] = p;
    stats[v].trackInsert();

    return p;
  }

  /// Get a reference to the table
  [[nodiscard]] const auto& getTables() const { return tables; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  /// Get a reference to individual statistics
  const UniqueTableStatistics& getStats(const std::size_t idx) const noexcept;

  /// Get a JSON object with the statistics
  nlohmann::basic_json<>
  getStatsJson(const bool includeIndividualTables = false) const;

  /// Get the total number of entries
  std::size_t getNumEntries() const noexcept;

  /// Get the total number of active entries
  std::size_t getNumActiveEntries() const noexcept;

  /// Get the peak total number of active entries
  std::size_t getPeakNumActiveEntries() const noexcept;

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
  bool incRef(NodeBase* p) noexcept;
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
  bool decRef(NodeBase* p) noexcept;

  bool possiblyNeedsCollection() const;

  std::size_t garbageCollect(bool force = false);

  void clear();

  template <class Node> void print() {
    auto q = cfg.nVars - 1U;
    for (auto it = tables.rbegin(); it != tables.rend(); ++it) {
      auto& table = *it;
      std::cout << "\tq" << q << ":"
                << "\n";
      for (std::size_t key = 0; key < table.size(); ++key) {
        auto* p = static_cast<Node*>(table[key]);
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
          p = p->next();
        }
      }
      --q;
    }
  }

private:
  /// Typedef for a bucket in the table
  using Bucket = NodeBase*;
  /// Typedef for the table
  using Table = std::vector<Bucket>;

  UniqueTableConfig cfg;

  /// The current garbage collection limit
  std::size_t gcLimit;

  /// A pointer to the memory manager for the nodes stored in the table.
  MemoryManager* memoryManager;

  /**
   * @brief The actual tables (one for each variable)
   * @details Each hash table is an array of buckets. Each bucket is a linked
   * list of entries. The linked list is implemented by using the next pointer
   * of the entries.
   */
  std::vector<Table> tables;

  /// A collection of statistics
  std::vector<UniqueTableStatistics> stats;

  /**
  Searches for a node in the hash table with the given key.
  @param p The node to search for.
  @param key The hashed value used to search the table.
  @return The Edge<Node> found in the hash table or Edge<Node>::zero if not
  found.
  **/
  template <class Node> Node* searchTable(Node* p, const std::size_t& key) {
    const auto v = p->v;
    Node* bucket = static_cast<Node*>(tables[v][key]);
    while (bucket != nullptr) {
      if (nodesAreEqual(p, bucket)) {
        // Match found
        if (p != bucket) {
          // put node pointed to by p on available chain
          memoryManager->returnEntry(p);
        }
        ++stats[v].hits;
        return bucket;
      }
      ++stats[v].collisions;
      bucket = bucket->next();
    }

    // Node not found in bucket
    return Node::getTerminal();
  }
};

} // namespace dd
