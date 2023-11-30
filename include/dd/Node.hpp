#pragma once

#include "dd/CachedEdge.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>

namespace dd {

/**
 * @brief A vector DD node
 * @details Data Layout |24|24|8|4|2| = 62B (space for two more bytes)
 */
struct vNode {                        // NOLINT(readability-identifier-naming)
  std::array<Edge<vNode>, RADIX> e{}; // edges out of this node
  vNode* next{};                      // used to link nodes in unique table
  RefCount ref{};                     // reference count
  Qubit v{};                          // variable index

  static constexpr bool isTerminal(const vNode* p) noexcept {
    return p == nullptr;
  }
  static constexpr vNode* getTerminal() noexcept { return nullptr; }
};
using vEdge = Edge<vNode>;
using vCachedEdge = CachedEdge<vNode>;

/**
 * @brief A matrix DD node
 * @details Data Layout |24|24|24|24|8|4|2|1| = 111B (space for one more byte)
 */
struct mNode {                        // NOLINT(readability-identifier-naming)
  std::array<Edge<mNode>, NEDGE> e{}; // edges out of this node
  mNode* next{};                      // used to link nodes in unique table
  RefCount ref{};                     // reference count
  Qubit v{};                          // variable index
  std::uint8_t flags = 0;
  // 32 = unused (was used to mark a node with is symmetric).
  // 16 = marks a node resembling identity
  // 8 = marks a reduced dm node,
  // 4 = marks a dm (tmp flag),
  // 2 = mark first path edge (tmp flag),
  // 1 = mark path is conjugated (tmp flag))

  [[nodiscard]] static constexpr bool isTerminal(const mNode* p) noexcept {
    return p == nullptr;
  }
  [[nodiscard]] static constexpr mNode* getTerminal() noexcept {
    return nullptr;
  }

  [[nodiscard]] inline bool isIdentity() const noexcept {
    return (flags & static_cast<std::uint8_t>(16U)) != 0;
  }
  [[nodiscard]] static constexpr bool isIdentity(const mNode* p) noexcept {
    return p == nullptr || p->isIdentity();
  }
  inline void setIdentity(const bool identity) noexcept {
    if (identity) {
      flags = (flags | static_cast<std::uint8_t>(16U));
    } else {
      flags = (flags & static_cast<std::uint8_t>(~16U));
    }
  }
};
using mEdge = Edge<mNode>;
using mCachedEdge = CachedEdge<mNode>;

/**
 * @brief A density matrix DD node
 * @details Data Layout |24|24|24|24|8|4|2|1| = 111B (space for one more byte)
 */
struct dNode {                        // NOLINT(readability-identifier-naming)
  std::array<Edge<dNode>, NEDGE> e{}; // edges out of this node
  dNode* next{};                      // used to link nodes in unique table
  RefCount ref{};                     // reference count
  Qubit v{};                          // variable index
  std::uint8_t flags = 0;
  // 32 = unused (was used to mark a node with is symmetric).
  // 16 = marks a node resembling identity
  // 8 = marks a reduced dm node,
  // 4 = marks a dm (tmp flag),
  // 2 = mark first path edge (tmp flag),
  // 1 = mark path is conjugated (tmp flag))

  static constexpr bool isTerminal(const dNode* p) noexcept {
    return p == nullptr;
  }
  static constexpr dNode* getTerminal() noexcept { return nullptr; }

  [[nodiscard]] static constexpr bool isIdentity(const dNode* p) noexcept {
    return p == nullptr;
  }

  [[nodiscard]] [[maybe_unused]] static constexpr bool
  tempDensityMatrixFlagsEqual(const std::uint8_t a,
                              const std::uint8_t b) noexcept {
    return getDensityMatrixTempFlags(a) == getDensityMatrixTempFlags(b);
  }

  [[nodiscard]] static constexpr bool
  isConjugateTempFlagSet(const std::uintptr_t p) noexcept {
    return (p & (1ULL << 0)) != 0U;
  }
  [[nodiscard]] static constexpr bool
  isNonReduceTempFlagSet(const std::uintptr_t p) noexcept {
    return (p & (1ULL << 1)) != 0U;
  }
  [[nodiscard]] static constexpr bool
  isDensityMatrixTempFlagSet(const std::uintptr_t p) noexcept {
    return (p & (1ULL << 2)) != 0U;
  }
  [[nodiscard]] static inline bool
  isDensityMatrixNode(const std::uintptr_t p) noexcept {
    return (p & (1ULL << 3)) != 0U;
  }

  [[nodiscard]] static bool isConjugateTempFlagSet(const dNode* p) noexcept {
    return isConjugateTempFlagSet(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] static bool isNonReduceTempFlagSet(const dNode* p) noexcept {
    return isNonReduceTempFlagSet(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] static bool
  isDensityMatrixTempFlagSet(const dNode* p) noexcept {
    return isDensityMatrixTempFlagSet(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] static bool isDensityMatrixNode(const dNode* p) noexcept {
    return isDensityMatrixNode(reinterpret_cast<std::uintptr_t>(p));
  }

  static void setConjugateTempFlagTrue(dNode*& p) noexcept {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) |
                                 (1ULL << 0));
  }
  static void setNonReduceTempFlagTrue(dNode*& p) noexcept {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) |
                                 (1ULL << 1));
  }
  static void setDensityMatTempFlagTrue(dNode*& p) noexcept {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) |
                                 (1ULL << 2));
  }
  static void alignDensityNode(dNode*& p) noexcept {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) & (~7ULL));
  }

  [[nodiscard]] static inline std::uintptr_t
  getDensityMatrixTempFlags(dNode*& p) noexcept {
    return getDensityMatrixTempFlags(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] static constexpr std::uintptr_t
  getDensityMatrixTempFlags(const std::uintptr_t a) noexcept {
    return a & (7ULL);
  }

  constexpr void unsetTempDensityMatrixFlags() noexcept {
    flags = flags & static_cast<std::uint8_t>(~7U);
  }

  void setDensityMatrixNodeFlag(bool densityMatrix) noexcept;

  static std::uint8_t alignDensityNodeNode(dNode*& p) noexcept;

  static void getAlignedNodeRevertModificationsOnSubEdges(dNode* p) noexcept;

  static void applyDmChangesToNode(dNode*& p) noexcept;

  static void revertDmChangesToNode(dNode*& p) noexcept;
};
using dEdge = Edge<dNode>;
using dCachedEdge = CachedEdge<dNode>;

static inline dEdge densityFromMatrixEdge(const mEdge& e) {
  return dEdge{reinterpret_cast<dNode*>(e.p), e.w};
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
  if (p == nullptr || p->ref == std::numeric_limits<RefCount>::max()) {
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
  if (p == nullptr || p->ref == std::numeric_limits<RefCount>::max()) {
    return false;
  }
  assert(p->ref != 0 &&
         "Reference count of Node must not be zero before decrement");
  --p->ref;
  return true;
}

} // namespace dd
