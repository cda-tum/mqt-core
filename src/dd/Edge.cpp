#include "dd/Edge.hpp"

#include "dd/Complex.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

namespace dd {

template <class Node> Edge<Node> Edge<Node>::terminal(const Complex& w) {
  return {Node::getTerminal(), w};
}

template <class Node> bool Edge<Node>::isTerminal() const {
  return Node::isTerminal(p);
}

template <class Node> bool Edge<Node>::isZeroTerminal() const {
  return isTerminal() && w == Complex::zero;
}

template <class Node> bool Edge<Node>::isOneTerminal() const {
  return isTerminal() && w == Complex::one;
}

template <class Node> bool Edge<Node>::isIdentity() const {
  if constexpr (std::is_same_v<Node, mNode> || std::is_same_v<Node, dNode>) {
    return isTerminal() && w != Complex::zero;
  }
  return false;
}

template <typename Node>
CachedEdge<Node>::CachedEdge(Node* n, const Complex& c) : p(n) {
  w.r = RealNumber::val(c.r);
  w.i = RealNumber::val(c.i);
}

template <class Node>
[[maybe_unused]] void Edge<Node>::setDensityConjugateTrue(Edge& e) {
  if constexpr (std::is_same_v<Node, dNode>) {
    Node::setConjugateTempFlagTrue(e.p);
  }
}

template <class Node>
[[maybe_unused]] void Edge<Node>::setFirstEdgeDensityPathTrue(Edge& e) {
  if constexpr (std::is_same_v<Node, dNode>) {
    Node::setNonReduceTempFlagTrue(e.p);
  }
}

template <class Node> void Edge<Node>::setDensityMatrixTrue(Edge& e) {
  if constexpr (std::is_same_v<Node, dNode>) {
    Node::setDensityMatTempFlagTrue(e.p);
  }
}

template <class Node> void Edge<Node>::alignDensityEdge(Edge& e) {
  if constexpr (std::is_same_v<Node, dNode>) {
    Node::alignDensityNode(e.p);
  }
}

template <class Node>
void Edge<Node>::revertDmChangesToEdges(Edge& x, Edge& y) {
  if constexpr (std::is_same_v<Node, dNode>) {
    revertDmChangesToEdge(x);
    revertDmChangesToEdge(y);
  }
}

template <class Node> void Edge<Node>::revertDmChangesToEdge(Edge& x) {
  if constexpr (std::is_same_v<Node, dNode>) {
    Node::revertDmChangesToNode(x.p);
  }
}

template <class Node> void Edge<Node>::applyDmChangesToEdges(Edge& x, Edge& y) {
  if constexpr (std::is_same_v<Node, dNode>) {
    applyDmChangesToEdge(x);
    applyDmChangesToEdge(y);
  }
}

template <class Node> void Edge<Node>::applyDmChangesToEdge(Edge& x) {
  if constexpr (std::is_same_v<Node, dNode>) {
    Node::applyDmChangesToNode(x.p);
  }
}

// Explicit instantiations
template struct Edge<vNode>;
template struct Edge<mNode>;
template struct Edge<dNode>;
template struct CachedEdge<vNode>;
template struct CachedEdge<mNode>;
template struct CachedEdge<dNode>;

} // namespace dd

namespace std {
template <class Node>
std::size_t
hash<dd::Edge<Node>>::operator()(const dd::Edge<Node>& e) const noexcept {
  const auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
  const auto h2 = std::hash<dd::Complex>{}(e.w);
  auto h3 = dd::combineHash(h1, h2);
  if constexpr (std::is_same_v<Node, dd::dNode>) {
    if (e.isTerminal()) {
      return h3;
    }
    assert((dd::dNode::isDensityMatrixTempFlagSet(e.p)) == false);
    const auto h4 = dd::dNode::getDensityMatrixTempFlags(e.p->flags);
    h3 = dd::combineHash(h3, h4);
  }
  return h3;
}

template <class Node>
std::size_t hash<dd::CachedEdge<Node>>::operator()(
    const dd::CachedEdge<Node>& e) const noexcept {
  const auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
  const auto h2 = std::hash<dd::ComplexValue>{}(e.w);
  return dd::combineHash(h1, h2);
}

template struct hash<dd::Edge<dd::vNode>>;
template struct hash<dd::Edge<dd::mNode>>;
template struct hash<dd::Edge<dd::dNode>>;
template struct hash<dd::CachedEdge<dd::vNode>>;
template struct hash<dd::CachedEdge<dd::mNode>>;
template struct hash<dd::CachedEdge<dd::dNode>>;
} // namespace std
