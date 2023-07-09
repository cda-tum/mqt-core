#pragma once

#include "Definitions.hpp"
#include "Edge.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

namespace dd {
// NOLINTNEXTLINE(readability-identifier-naming)
struct vNode {
  std::array<Edge<vNode>, RADIX> e{}; // edges out of this node
  vNode* next{};                      // used to link nodes in unique table
  RefCount ref{};                     // reference count
  Qubit v{}; // variable index (nonterminal) value (-1 for terminal)

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables
  static vNode terminal;

  static constexpr bool isTerminal(const vNode* p) { return p == &terminal; }
  static constexpr vNode* getTerminal() { return &terminal; }
};
using vEdge = Edge<vNode>;
using vCachedEdge = CachedEdge<vNode>;

// NOLINTNEXTLINE(readability-identifier-naming)
struct mNode {
  std::array<Edge<mNode>, NEDGE> e{}; // edges out of this node
  mNode* next{};                      // used to link nodes in unique table
  RefCount ref{};                     // reference count
  Qubit v{}; // variable index (nonterminal) value (-1 for terminal)
  std::uint8_t flags = 0;
  // 32 = marks a node with is symmetric.
  // 16 = marks a node resembling identity
  // 8 = marks a reduced dm node,
  // 4 = marks a dm (tmp flag),
  // 2 = mark first path edge (tmp flag),
  // 1 = mark path is conjugated (tmp flag))

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static mNode terminal;

  static constexpr bool isTerminal(const mNode* p) { return p == &terminal; }
  static constexpr mNode* getTerminal() { return &terminal; }

  [[nodiscard]] inline bool isIdentity() const {
    return (flags & static_cast<std::uint8_t>(16U)) != 0;
  }
  [[nodiscard]] inline bool isSymmetric() const {
    return (flags & static_cast<std::uint8_t>(32U)) != 0;
  }

  inline void setIdentity(bool identity) {
    if (identity) {
      flags = (flags | static_cast<std::uint8_t>(16U));
    } else {
      flags = (flags & static_cast<std::uint8_t>(~16U));
    }
  }
  inline void setSymmetric(bool symmetric) {
    if (symmetric) {
      flags = (flags | static_cast<std::uint8_t>(32U));
    } else {
      flags = (flags & static_cast<std::uint8_t>(~32U));
    }
  }
};
using mEdge = Edge<mNode>;
using mCachedEdge = CachedEdge<mNode>;

// NOLINTNEXTLINE(readability-identifier-naming)
struct dNode {
  std::array<Edge<dNode>, NEDGE> e{}; // edges out of this node
  dNode* next{};                      // used to link nodes in unique table
  RefCount ref{};                     // reference count
  Qubit v{}; // variable index (nonterminal) value (-1 for terminal)
  std::uint8_t flags = 0;
  // 32 = marks a node with is symmetric.
  // 16 = marks a node resembling identity
  // 8 = marks a reduced dm node,
  // 4 = marks a dm (tmp flag),
  // 2 = mark first path edge (tmp flag),
  // 1 = mark path is conjugated (tmp flag))

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static dNode terminal;

  static constexpr bool isTerminal(const dNode* p) { return p == &terminal; }
  static constexpr dNode* getTerminal() { return &terminal; }

  [[nodiscard]] [[maybe_unused]] static inline bool
  tempDensityMatrixFlagsEqual(const std::uint8_t a, const std::uint8_t b) {
    return getDensityMatrixTempFlags(a) == getDensityMatrixTempFlags(b);
  }

  [[nodiscard]] static inline bool
  isConjugateTempFlagSet(const std::uintptr_t p) {
    return (p & (1ULL << 0)) != 0U;
  }
  [[nodiscard]] static inline bool
  isNonReduceTempFlagSet(const std::uintptr_t p) {
    return (p & (1ULL << 1)) != 0U;
  }
  [[nodiscard]] static inline bool
  isDensityMatrixTempFlagSet(const std::uintptr_t p) {
    return (p & (1ULL << 2)) != 0U;
  }
  [[nodiscard]] static inline bool isDensityMatrixNode(const std::uintptr_t p) {
    return (p & (1ULL << 3)) != 0U;
  }

  [[nodiscard]] [[maybe_unused]] static inline bool
  isConjugateTempFlagSet(const dNode* p) {
    return isConjugateTempFlagSet(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] [[maybe_unused]] static inline bool
  isNonReduceTempFlagSet(const dNode* p) {
    return isNonReduceTempFlagSet(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] [[maybe_unused]] static inline bool
  isDensityMatrixTempFlagSet(const dNode* p) {
    return isDensityMatrixTempFlagSet(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] [[maybe_unused]] static inline bool
  isDensityMatrixNode(const dNode* p) {
    return isDensityMatrixNode(reinterpret_cast<std::uintptr_t>(p));
  }

  static inline void setConjugateTempFlagTrue(dNode*& p) {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) |
                                 (1ULL << 0));
  }
  static inline void setNonReduceTempFlagTrue(dNode*& p) {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) |
                                 (1ULL << 1));
  }
  static inline void setDensityMatTempFlagTrue(dNode*& p) {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) |
                                 (1ULL << 2));
  }
  static inline void alignDensityNode(dNode*& p) {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) & (~7ULL));
  }

  [[nodiscard]] static inline std::uintptr_t
  getDensityMatrixTempFlags(dNode*& p) {
    return getDensityMatrixTempFlags(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] static inline std::uintptr_t
  getDensityMatrixTempFlags(const std::uintptr_t a) {
    return a & (7ULL);
  }

  void unsetTempDensityMatrixFlags() {
    flags = flags & static_cast<std::uint8_t>(~7U);
  }

  void setDensityMatrixNodeFlag(bool densityMatrix);

  static std::uint8_t alignDensityNodeNode(dNode*& p);

  static void getAlignedNodeRevertModificationsOnSubEdges(dNode* p);

  static void applyDmChangesToNode(dNode*& p);

  static void revertDmChangesToNode(dNode*& p);
};
using dEdge = Edge<dNode>;
using dCachedEdge = CachedEdge<dNode>;

static inline dEdge densityFromMatrixEdge(const mEdge& e) {
  return dEdge{reinterpret_cast<dNode*>(e.p), e.w};
}

/**
 * @brief Indicates whether a given node needs reference count updates.
 * @details This function checks whether a given node needs reference count
 * updates. A node needs reference count updates if the pointer to it is
 * not the null pointer, if it is not a terminal node, and if the reference
 * count has saturated.
 * @tparam Node Type of the node to check.
 * @param num Pointer to the node to check.
 * @returns Whether the node needs reference count updates.
 */
template <typename Node>
[[nodiscard]] static inline bool
noRefCountingNeeded(const Node* const p) noexcept {
  return p == nullptr || Node::isTerminal(p) ||
         p->ref == std::numeric_limits<RefCount>::max();
}

/**
 * @brief Increment the reference count of a node.
 * @details This function increments the reference count of a node. If the
 * reference count has saturated (i.e. reached the maximum value of RefCount)
 * the reference count is not incremented.
 * @tparam Node Type of the node to increment the reference count of.
 * @param p A pointer to the node to increment the reference count of.
 * @returns Whether the reference count was incremented.
 * @note Typically, you do not want to call this function directly. Instead,
 * use the UniqueTable::incRef(Node*) function.
 */
template <typename Node>
[[nodiscard]] static inline bool incRef(Node* p) noexcept {
  if (noRefCountingNeeded(p)) {
    return false;
  }
  ++p->ref;
  return true;
}

/**
 * @brief Decrement the reference count of a node.
 * @details This function decrements the reference count of a node. If the
 * reference count has saturated (i.e. reached the maximum value of RefCount)
 * the reference count is not decremented.
 * @tparam Node Type of the node to decrement the reference count of.
 * @param p A pointer to the node to decrement the reference count of.
 * @returns Whether the reference count was decremented.
 * @note Typically, you do not want to call this function directly. Instead,
 * use the UniqueTable::decRef(Node*) function.
 */
template <typename Node>
[[nodiscard]] static inline bool decRef(Node* p) noexcept {
  if (noRefCountingNeeded(p)) {
    return false;
  }
  assert(p->ref != 0 &&
         "Reference count of Node must not be zero before decrement");
  --p->ref;
  return true;
}

} // namespace dd
