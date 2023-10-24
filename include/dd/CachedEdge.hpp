#pragma once

#include "dd/Complex.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"

#include <complex>
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
  CachedEdge(Node* n, const Complex& c)
      : p(n), w(static_cast<ComplexValue>(c)) {}

  /// Comparing two DD edges with another involves comparing the respective
  /// pointers and checking whether the corresponding weights are "close enough"
  /// according to a given tolerance this notion of equivalence is chosen to
  /// counter floating point inaccuracies
  bool operator==(const CachedEdge& other) const {
    return p == other.p && w.approximatelyEquals(other.w);
  }
  bool operator!=(const CachedEdge& other) const { return !operator==(other); }

  /**
   * @brief Create a terminal edge with the given weight.
   * @param w The weight of the terminal edge.
   * @return A terminal edge with the given weight.
   */
  [[nodiscard]] static constexpr CachedEdge terminal(const ComplexValue& w) {
    return CachedEdge{Node::getTerminal(), w};
  }

  /**
   * @brief Create a terminal edge with the given weight.
   * @param w The weight of the terminal edge.
   * @return A terminal edge with the given weight.
   */
  [[nodiscard]] static constexpr CachedEdge
  terminal(const std::complex<fp>& w) {
    return CachedEdge{Node::getTerminal(), static_cast<ComplexValue>(w)};
  }

  /**
   * @brief Create a terminal edge with the given weight.
   * @param w The weight of the terminal edge.
   * @return A terminal edge with the given weight.
   */
  [[nodiscard]] static constexpr CachedEdge terminal(const Complex& w) {
    return terminal(static_cast<ComplexValue>(w));
  }

  /**
   * @brief Create a zero terminal edge.
   * @return A zero terminal edge.
   */
  [[nodiscard]] static constexpr CachedEdge zero() {
    return terminal(ComplexValue(0.));
  }

  /**
   * @brief Create a one terminal edge.
   * @return A one terminal edge.
   */
  [[nodiscard]] static constexpr CachedEdge one() {
    return terminal(ComplexValue(1.));
  }
};

} // namespace dd

namespace std {
template <class Node> struct hash<dd::CachedEdge<Node>> {
  std::size_t operator()(dd::CachedEdge<Node> const& e) const noexcept;
};
} // namespace std
