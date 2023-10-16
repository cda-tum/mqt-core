#include "dd/CachedEdge.hpp"

#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

#include <algorithm>
#include <array>
#include <cassert>
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

  const auto mag0 = e[0].w.mag2();
  const auto mag1 = e[1].w.mag2();
  const auto norm2 = mag0 + mag1;
  const auto mag2Max = (mag0 + RealNumber::eps >= mag1) ? mag0 : mag1;
  const auto argMax = (mag0 + RealNumber::eps >= mag1) ? 0U : 1U;
  const auto argMin = (argMax + 1U) % 2U;
  const auto norm = std::sqrt(norm2);
  const auto magMax = std::sqrt(mag2Max);
  const auto commonFactor = norm / magMax;

  const auto& max = e[argMax];
  const auto topWeight = max.w * commonFactor;

  const auto maxWeight = cn.lookup(magMax / norm);
  if (maxWeight.exactlyZero()) {
    p->e[argMax] = vEdge::zero();
  } else {
    p->e[argMax] = {max.p, maxWeight};
  }

  const auto& min = e[argMin];
  const auto minWeight = cn.lookup(min.w / topWeight);
  if (minWeight.exactlyZero()) {
    p->e[argMin] = vEdge::zero();
  } else {
    p->e[argMin] = {min.p, minWeight};
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

  std::optional<std::size_t> argMax = std::nullopt;
  fp maxMag = 0.;
  ComplexValue maxVal = 1.;
  // determine max amplitude
  for (auto i = 0U; i < NEDGE; ++i) {
    if (zero[i]) {
      continue;
    }
    const auto& w = e[i].w;
    if (!argMax.has_value()) {
      argMax = i;
      maxMag = w.mag2();
      maxVal = w;
    } else {
      if (const auto mag = w.mag2(); mag - maxMag > RealNumber::eps) {
        argMax = i;
        maxMag = mag;
        maxVal = w;
      }
    }
  }
  assert(argMax.has_value() && "argMax should have been set by now");

  const auto argMaxValue = *argMax;
  for (auto i = 0U; i < NEDGE; ++i) {
    auto& successor = p->e[i];
    successor.p = e[i].p;
    if (i == argMaxValue) {
      successor.w = Complex::one();
      continue;
    }
    const auto& weight = e[i].w;
    // The approximation below is really important for numerical stability.
    // An exactly zero check will lead to numerical instabilities.
    if (zero[i]) {
      successor = Edge<Node>::zero();
      continue;
    }
    successor.w = cn.lookup(weight / maxVal);
    if (successor.w.exactlyZero()) {
      successor.p = Node::getTerminal();
    }
  }
  return CachedEdge<Node>{p, maxVal};
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
  const auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
  const auto h2 = std::hash<dd::ComplexValue>{}(e.w);
  return dd::combineHash(h1, h2);
}

template struct hash<dd::CachedEdge<dd::vNode>>;
template struct hash<dd::CachedEdge<dd::mNode>>;
template struct hash<dd::CachedEdge<dd::dNode>>;
} // namespace std
