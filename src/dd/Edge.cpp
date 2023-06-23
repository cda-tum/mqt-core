#include "dd/Edge.hpp"

#include "dd/Node.hpp"

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

template <class Node>
const Edge<Node> Edge<Node>::zero{Node::getTerminal(), Complex::zero};
template <class Node>
const Edge<Node> Edge<Node>::one{Node::getTerminal(), Complex::one};

// Explicit instantiations
template struct Edge<vNode>;
template struct Edge<mNode>;
template struct Edge<dNode>;

} // namespace dd
