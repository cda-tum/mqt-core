#include "dd/CachedEdge.hpp"

#include "Definitions.hpp"
#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <optional>

namespace dd {

///-----------------------------------------------------------------------------
///                      \n Methods for vector DDs \n
///-----------------------------------------------------------------------------

template <class Node>
template <typename T, isVector<T>>
CachedEdge<Node>
CachedEdge<Node>::normalize(Node* p,
                            const std::array<CachedEdge<Node>, RADIX>& e,
                            MemoryManager<Node>& mm, ComplexNumbers& cn) {
  assert(p != nullptr && "Node pointer passed to normalize is null.");
  const auto zero =
      std::array{e[0].w.approximatelyZero(), e[1].w.approximatelyZero()};

  if (zero[0]) {
    if (zero[1]) {
      mm.returnEntry(p);
      return CachedEdge::zero();
    }
    p->e = {vEdge::zero(), {e[1].p, Complex::one()}};
    return {p, e[1].w};
  }

  if (zero[1]) {
    p->e = {vEdge{e[0].p, Complex::one()}, vEdge::zero()};
    return {p, e[0].w};
  }

  const auto mag2 = std::array{e[0].w.mag2(), e[1].w.mag2()};

  const auto argMax = (mag2[0] + RealNumber::eps >= mag2[1]) ? 0U : 1U;
  const auto& maxMag2 = mag2[argMax];

  const auto argMin = 1U - argMax;
  const auto& minMag2 = mag2[argMin];

  const auto norm = std::sqrt(maxMag2 + minMag2);
  const auto maxMag = std::sqrt(maxMag2);
  const auto commonFactor = norm / maxMag;

  const auto topWeight = e[argMax].w * commonFactor;
  const auto maxWeight = maxMag / norm;
  const auto minWeight = e[argMin].w / topWeight;

  p->e[argMax] = {e[argMax].p, cn.lookup(maxWeight)};
  assert(!p->e[argMax].w.exactlyZero() &&
         "Max edge weight should not be zero.");

  const auto minW = cn.lookup(minWeight);
  if (minW.exactlyZero()) {
    assert(p->e[argMax].w.exactlyOne() &&
           "Edge weight should be one when minWeight is zero.");
    p->e[argMin] = vEdge::zero();
  } else {
    p->e[argMin] = {e[argMin].p, minW};
  }

  return {p, topWeight};
}

///-----------------------------------------------------------------------------
///                      \n Methods for matrix DDs \n
///-----------------------------------------------------------------------------

template <class Node>
template <typename T, isMatrixVariant<T>>
CachedEdge<Node>
CachedEdge<Node>::normalize(Node* p,
                            const std::array<CachedEdge<Node>, NEDGE>& e,
                            MemoryManager<Node>& mm, ComplexNumbers& cn) {
  assert(p != nullptr && "Node pointer passed to normalize is null.");
  const auto zero =
      std::array{e[0].w.approximatelyZero(), e[1].w.approximatelyZero(),
                 e[2].w.approximatelyZero(), e[3].w.approximatelyZero()};

  if (std::all_of(zero.begin(), zero.end(), [](auto b) { return b; })) {
    mm.returnEntry(p);
    return CachedEdge::zero();
  }

  const auto mag2 =
      std::array{e[0].w.mag2(), e[1].w.mag2(), e[2].w.mag2(), e[3].w.mag2()};

  std::size_t argMax = 0U;
  auto maxMag2 = mag2[0];
  for (std::size_t i = 1U; i < NEDGE; ++i) {
    if (const auto mag2i = mag2[i]; mag2i + RealNumber::eps > maxMag2) {
      argMax = i;
      maxMag2 = mag2i;
    }
  }

  // pair up 0 <-> 2 and 1 <-> 3
  const auto argMin = (argMax + 2U) % 4;
  const auto minMag2 = mag2[argMin];

  const auto norm = std::sqrt(maxMag2 + minMag2);
  const auto maxMag = std::sqrt(maxMag2);
  const auto commonFactor = norm / maxMag;

  const auto topWeight = e[argMax].w * commonFactor;
  const auto maxWeight = maxMag / norm;
  p->e[argMax] = {e[argMax].p, cn.lookup(maxWeight)};
  assert(!p->e[argMax].w.exactlyZero() &&
         "Max edge weight should not be zero.");

  for (std::size_t i = 0; i < NEDGE; ++i) {
    if (i == argMax) {
      continue;
    }
    const auto weight = e[i].w / topWeight;
    auto& successor = p->e[i];
    successor.p = e[i].p;
    successor.w = cn.lookup(weight);
    if (successor.w.exactlyZero()) {
      successor.p = Node::getTerminal();
    }
  }
  return {p, topWeight};
}

///-----------------------------------------------------------------------------
///                      \n Explicit instantiations \n
///-----------------------------------------------------------------------------

template struct CachedEdge<vNode>;
template struct CachedEdge<mNode>;
template struct CachedEdge<dNode>;

template CachedEdge<vNode>
CachedEdge<vNode>::normalize(vNode* p,
                             const std::array<CachedEdge<vNode>, RADIX>& e,
                             MemoryManager<vNode>& mm, ComplexNumbers& cn);

template CachedEdge<mNode>
CachedEdge<mNode>::normalize(mNode* p,
                             const std::array<CachedEdge<mNode>, NEDGE>& e,
                             MemoryManager<mNode>& mm, ComplexNumbers& cn);

template CachedEdge<dNode>
CachedEdge<dNode>::normalize(dNode* p,
                             const std::array<CachedEdge<dNode>, NEDGE>& e,
                             MemoryManager<dNode>& mm, ComplexNumbers& cn);

} // namespace dd

namespace std {
template <class Node>
std::size_t hash<dd::CachedEdge<Node>>::operator()(
    const dd::CachedEdge<Node>& e) const noexcept {
  const auto h1 = qc::murmur64(reinterpret_cast<std::size_t>(e.p));
  const auto h2 = std::hash<dd::ComplexValue>{}(e.w);
  return qc::combineHash(h1, h2);
}

template struct hash<dd::CachedEdge<dd::vNode>>;
template struct hash<dd::CachedEdge<dd::mNode>>;
template struct hash<dd::CachedEdge<dd::dNode>>;
} // namespace std
