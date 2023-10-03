#pragma once

#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"

#include <utility>

namespace dd {

///-----------------------------------------------------------------------------
///                        \n Forward declarations \n
///-----------------------------------------------------------------------------
struct Complex;

/**
 * @brief A DD node with a cached edge weight
 * @details Some DD operations create intermediate results that are not part of
 * the final result. To avoid storing these intermediate results in the unique
 * table, they are represented via cached numbers.
 * @tparam Node Type of the DD node
 */
template <typename Node> struct CachedEdge {
  Node* p{};
  ComplexValue w{};

  CachedEdge() = default;
  CachedEdge(Node* n, const ComplexValue& v) : p(n), w(v) {}
  CachedEdge(Node* n, const Complex& c);

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
template <class Node> struct hash<dd::CachedEdge<Node>> {
  std::size_t operator()(dd::CachedEdge<Node> const& e) const noexcept;
};
} // namespace std
