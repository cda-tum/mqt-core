#pragma once

#include "Complex.hpp"
#include "ComplexValue.hpp"
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

} // namespace dd

namespace std {
template <> struct hash<dd::dEdge> {
  std::size_t operator()(dd::dEdge const& e) const noexcept {
    const auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
    const auto h2 = std::hash<dd::Complex>{}(e.w);
    assert(e.p != nullptr);
    assert((dd::dNode::isDensityMatrixTempFlagSet(e.p)) == false);
    const auto h3 = std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(
        dd::dNode::getDensityMatrixTempFlags(e.p->flags)));
    const auto tmp = dd::combineHash(h1, h2);
    return dd::combineHash(tmp, h3);
  }
};
} // namespace std
