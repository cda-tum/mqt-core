#include "dd/CachedEdge.hpp"

#include "dd/Complex.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

namespace dd {

template <typename Node>
CachedEdge<Node>::CachedEdge(Node* n, const Complex& c) : p(n) {
  w.r = RealNumber::val(c.r);
  w.i = RealNumber::val(c.i);
}

template struct CachedEdge<vNode>;
template struct CachedEdge<mNode>;
template struct CachedEdge<dNode>;

} // namespace dd

namespace std {
template <class Node>
std::size_t hash<dd::CachedEdge<Node>>::operator()(
    const dd::CachedEdge<Node>& e) const noexcept {
  const auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
  const auto h2 = std::hash<dd::ComplexValue>{}(e.w);
  return dd::combineHash(h1, h2);
}

template struct hash<dd::CachedEdge<dd::vNode>>;
template struct hash<dd::CachedEdge<dd::mNode>>;
template struct hash<dd::CachedEdge<dd::dNode>>;
} // namespace std
