#pragma once

#include "Complex.hpp"
#include "ComplexValue.hpp"
#include "Definitions.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace dd {
template <class Node> struct Edge {
  Node* p;
  Complex w;

  /// Comparing two DD edges with another involves comparing the respective
  /// pointers and checking whether the corresponding weights are "close enough"
  /// according to a given tolerance this notion of equivalence is chosen to
  /// counter floating point inaccuracies
  constexpr bool operator==(const Edge& other) const {
    return p == other.p && w.approximatelyEquals(other.w);
  }
  constexpr bool operator!=(const Edge& other) const {
    return !operator==(other);
  }

  // edges pointing to zero and one terminals
  static const Edge zero; // NOLINT(readability-identifier-naming)
  static const Edge one;  // NOLINT(readability-identifier-naming)

  [[nodiscard]] static Edge terminal(const Complex& w);
  [[nodiscard]] bool isTerminal() const;
  [[nodiscard]] bool isZeroTerminal() const;
  [[nodiscard]] bool isOneTerminal() const;

  // Functions only related to density matrices
  [[maybe_unused]] static void setDensityConjugateTrue(Edge& e);
  [[maybe_unused]] static void setFirstEdgeDensityPathTrue(Edge& e);
  static void setDensityMatrixTrue(Edge& e);
  static void alignDensityEdge(Edge& e);
  static void revertDmChangesToEdges(Edge& x, Edge& y);
  static void revertDmChangesToEdge(Edge& x);
  static void applyDmChangesToEdges(Edge& x, Edge& y);
  static void applyDmChangesToEdge(Edge& x);
};

template <class Node>
const Edge<Node> Edge<Node>::zero{Node::getTerminal(), Complex::zero};
template <class Node>
const Edge<Node> Edge<Node>::one{Node::getTerminal(), Complex::one};

template <typename Node> struct CachedEdge {
  Node* p{};
  ComplexValue w{};

  CachedEdge() = default;
  CachedEdge(Node* n, const ComplexValue& v) : p(n), w(v) {}
  CachedEdge(Node* n, const Complex& c) : p(n) {
    w.r = CTEntry::val(c.r);
    w.i = CTEntry::val(c.i);
  }

  /// Comparing two DD edges with another involves comparing the respective
  /// pointers and checking whether the corresponding weights are "close enough"
  /// according to a given tolerance this notion of equivalence is chosen to
  /// counter floating point inaccuracies
  bool operator==(const CachedEdge& other) const {
    return p == other.p && w.approximatelyEquals(other.w);
  }
  bool operator!=(const CachedEdge& other) const { return !operator==(other); }
};
} // namespace dd

namespace std {
template <class Node> struct hash<dd::Edge<Node>> {
  std::size_t operator()(dd::Edge<Node> const& e) const noexcept {
    auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
    auto h2 = std::hash<dd::Complex>{}(e.w);
    return dd::combineHash(h1, h2);
  }
};

template <class Node> struct hash<dd::CachedEdge<Node>> {
  std::size_t operator()(dd::CachedEdge<Node> const& e) const noexcept {
    auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
    auto h2 = std::hash<dd::ComplexValue>{}(e.w);
    return dd::combineHash(h1, h2);
  }
};
} // namespace std
