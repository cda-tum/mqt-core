#include "dd/Edge.hpp"

#include "dd/Node.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace dd {

///-----------------------------------------------------------------------------
///                      \n General purpose methods \n
///-----------------------------------------------------------------------------

template <class Node>
std::complex<fp>
Edge<Node>::getValueByPath(const std::string& decisions) const {
  if (isTerminal()) {
    return static_cast<std::complex<fp>>(w);
  }

  auto c = std::complex<fp>{1.0, 0.0};
  auto r = *this;
  if constexpr (std::is_same_v<Node, dNode>) {
    Edge<dNode>::applyDmChangesToEdge(r);
  }
  while (!r.isTerminal()) {
    c *= r.w;
    const auto tmp = static_cast<std::size_t>(decisions.at(r.p->v) - '0');
    assert(tmp <= r.p->e.size());

    if constexpr (std::is_same_v<Node, dNode>) {
      auto e = r;
      Edge<dNode>::applyDmChangesToEdge(r.p->e[tmp]);
      r = r.p->e[tmp];
      Edge<dNode>::revertDmChangesToEdge(e);
    } else {
      r = r.p->e[tmp];
    }
  }
  c *= r.w;
  return c;
}

template <class Node> std::size_t Edge<Node>::size() const {
  static constexpr std::size_t NODECOUNT_BUCKETS = 200000U;
  static std::unordered_set<const Node*> visited{NODECOUNT_BUCKETS};
  visited.max_load_factor(10);
  visited.clear();
  return size(visited);
}

template <class Node>
std::size_t Edge<Node>::size(std::unordered_set<const Node*>& visited) const {
  visited.emplace(p);
  std::size_t sum = 1U;
  if (!isTerminal()) {
    for (const auto& e : p->e) {
      if (visited.find(e.p) == visited.end()) {
        sum += e.size(visited);
      }
    }
  }
  return sum;
}

///-----------------------------------------------------------------------------
///                      \n Methods for vector DDs \n
///-----------------------------------------------------------------------------

template <class Node>
template <typename T, isVector<T>>
std::complex<fp> Edge<Node>::getValueByIndex(const std::size_t i) const {
  if (isTerminal()) {
    return w;
  }

  auto decisions = std::string(p->v + 1U, '0');
  for (auto j = 0U; j <= p->v; ++j) {
    if ((i & (1U << j)) != 0U) {
      decisions[j] = '1';
    }
  }

  return getValueByPath(decisions);
}

template <class Node>
template <typename T, isVector<T>>
CVec Edge<Node>::getVector(const fp threshold) const {
  if (isTerminal()) {
    return {w};
  }

  const std::size_t dim = 2ULL << p->v;
  auto vec = CVec(dim, 0.);
  traverseVector(
      1., 0,
      [&vec](const std::size_t i, const std::complex<fp>& c) { vec.at(i) = c; },
      threshold);
  return vec;
}

template <class Node>
template <typename T, isVector<T>>
SparseCVec Edge<Node>::getSparseVector(const fp threshold) const {
  if (isTerminal()) {
    return {{0, w}};
  }

  auto vec = SparseCVec{};
  traverseVector(
      1., 0,
      [&vec](const std::size_t i, const std::complex<fp>& c) { vec[i] = c; },
      threshold);
  return vec;
}

template <class Node>
template <typename T, isVector<T>>
void Edge<Node>::printVector() const {
  constexpr auto precision = 3;
  const auto oldPrecision = std::cout.precision();
  std::cout << std::setprecision(precision);

  if (isTerminal()) {
    std::cout << "0: " << w << "\n";
    return;
  }
  const std::size_t element = 2ULL << p->v;
  for (auto i = 0ULL; i < element; i++) {
    const auto amplitude = getValueByIndex(i);
    const auto n = static_cast<std::size_t>(p->v) + 1U;
    for (auto j = n; j > 0; --j) {
      std::cout << ((i >> (j - 1)) & 1ULL);
    }
    std::cout << ": " << amplitude << "\n";
  }
  std::cout << std::setprecision(static_cast<int>(oldPrecision));
  std::cout << std::flush;
}

template <class Node>
template <typename T, isVector<T>>
void Edge<Node>::addToVector(dd::CVec& amplitudes) const {
  if (isTerminal()) {
    amplitudes[0] += w;
    return;
  }

  traverseVector(1., 0,
                 [&amplitudes](const std::size_t i, const std::complex<fp>& c) {
                   amplitudes[i] += c;
                 });
}

template <class Node>
template <typename T, isVector<T>>
void Edge<Node>::traverseVector(const std::complex<fp>& amp,
                                const std::size_t i, AmplitudeFunc f,
                                const fp threshold) const {
  // calculate new accumulated amplitude
  const auto c = amp * static_cast<std::complex<fp>>(w);

  if (std::abs(c) < threshold) {
    return;
  }

  if (isTerminal()) {
    f(i, c);
    return;
  }

  // recursive case
  if (const auto& e = p->e[0]; e.w != 0.) {
    e.traverseVector(c, i, f, threshold);
  }
  if (const auto& e = p->e[1]; e.w != 0.) {
    e.traverseVector(c, i | (1ULL << p->v), f, threshold);
  }
}

///-----------------------------------------------------------------------------
///                      \n Methods for matrix DDs \n
///-----------------------------------------------------------------------------

template <class Node>
template <typename T, isMatrixVariant<T>>
std::complex<fp> Edge<Node>::getValueByIndex(const std::size_t i,
                                             const std::size_t j) const {
  if (isTerminal()) {
    return w;
  }

  auto decisions = std::string(p->v + 1U, '0');
  for (auto k = 0U; k <= p->v; ++k) {
    if ((i & (1U << k)) != 0U) {
      decisions[k] = '2';
    }
  }
  for (auto k = 0U; k <= p->v; ++k) {
    if ((j & (1U << k)) != 0U) {
      if (decisions[k] == '2') {
        decisions[k] = '3';
      } else {
        decisions[k] = '1';
      }
    }
  }

  return getValueByPath(decisions);
}

template <class Node>
template <typename T, isMatrixVariant<T>>
CMat Edge<Node>::getMatrix(const fp threshold) const {
  if (isTerminal()) {
    return CMat{1, {w}};
  }

  auto r = *this;
  if constexpr (std::is_same_v<Node, dNode>) {
    Edge<dNode>::applyDmChangesToEdge(r);
  }

  const std::size_t dim = 2ULL << r.p->v;
  auto mat = CMat(dim, CVec(dim, 0.0));

  r.traverseMatrix(
      1, 0ULL, 0ULL,
      [&mat](const std::size_t i, const std::size_t j,
             const std::complex<fp>& c) { mat.at(i).at(j) = c; },
      threshold);

  if constexpr (std::is_same_v<Node, dNode>) {
    Edge<dNode>::revertDmChangesToEdge(r);
  }
  return mat;
}

template <class Node>
template <typename T, isMatrixVariant<T>>
SparseCMat Edge<Node>::getSparseMatrix(const fp threshold) const {
  if (isTerminal()) {
    return {{{0U, 0U}, w}};
  }

  auto r = *this;
  if constexpr (std::is_same_v<Node, dNode>) {
    Edge<dNode>::applyDmChangesToEdge(r);
  }

  auto mat = SparseCMat{};
  r.traverseMatrix(
      1, 0ULL, 0ULL,
      [&mat](const std::size_t i, const std::size_t j,
             const std::complex<fp>& c) {
        mat[{i, j}] = c;
      },
      threshold);

  if constexpr (std::is_same_v<Node, dNode>) {
    Edge<dNode>::revertDmChangesToEdge(r);
  }

  return mat;
}

template <class Node>
template <typename T, isMatrixVariant<T>>
void Edge<Node>::printMatrix() const {
  constexpr auto precision = 3;
  const auto oldPrecision = std::cout.precision();
  std::cout << std::setprecision(precision);

  if (isTerminal()) {
    std::cout << w << "\n";
    return;
  }

  auto r = *this;
  if constexpr (std::is_same_v<Node, dNode>) {
    Edge<dNode>::alignDensityEdge(r);
  }

  const std::size_t element = 2ULL << r.p->v;
  for (auto i = 0ULL; i < element; ++i) {
    for (auto j = 0ULL; j < element; ++j) {
      const auto amplitude = getValueByIndex(i, j);
      std::cout << amplitude << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::setprecision(static_cast<int>(oldPrecision));
  std::cout << std::flush;
}

template <class Node>
template <typename T, isMatrixVariant<T>>
void Edge<Node>::traverseMatrix(const std::complex<fp>& amp,
                                const std::size_t i, const std::size_t j,
                                MatrixEntryFunc f, const fp threshold) const {
  // calculate new accumulated amplitude
  const auto c = amp * w;

  if (std::abs(c) < threshold) {
    return;
  }

  if (isTerminal()) {
    f(i, j, c);
    return;
  }

  const std::size_t x = i | (1ULL << p->v);
  const std::size_t y = j | (1ULL << p->v);
  const auto coords = {std::pair{i, j}, {i, y}, {x, j}, {x, y}};
  std::size_t k = 0U;
  for (const auto& [a, b] : coords) {
    if (auto& e = p->e[k++]; e.w != 0.) {
      if constexpr (std::is_same_v<Node, dNode>) {
        Edge<dNode>::applyDmChangesToEdge(e);
      }
      e.traverseMatrix(c, a, b, f, threshold);
      if constexpr (std::is_same_v<Node, dNode>) {
        Edge<dNode>::revertDmChangesToEdge(e);
      }
    }
  }
}

///-----------------------------------------------------------------------------
///                   \n Methods for density matrix DDs \n
///-----------------------------------------------------------------------------

template <class Node>
template <typename T, isDensityMatrix<T>>
SparsePVec Edge<Node>::getSparseProbabilityVector(const fp threshold) const {
  if (isTerminal()) {
    return {{0, w.real()}};
  }

  auto e = *this;
  Edge<dNode>::alignDensityEdge(e);

  auto probabilities = SparsePVec{};
  e.traverseDiagonal(
      1, 0,
      [&probabilities](const std::size_t i, const fp& prob) {
        probabilities[i] = prob;
      },
      threshold);
  return probabilities;
}

template <class Node>
template <typename T, isDensityMatrix<T>>
SparsePVecStrKeys
Edge<Node>::getSparseProbabilityVectorStrKeys(const fp threshold) const {
  if (isTerminal()) {
    return {{"0", w.real()}};
  }

  auto e = *this;
  Edge<dNode>::alignDensityEdge(e);
  const auto nqubits = static_cast<std::size_t>(e.p->v) + 1U;

  auto probabilities = SparsePVecStrKeys{};
  e.traverseDiagonal(
      1, 0,
      [&probabilities, &nqubits](const std::size_t i, const fp& prob) {
        probabilities[intToBinaryString(i, nqubits)] = prob;
      },
      threshold);
  return probabilities;
}

template <class Node>
template <typename T, isDensityMatrix<T>>
void Edge<Node>::traverseDiagonal(const fp& prob, const std::size_t i,
                                  ProbabilityFunc f,
                                  const dd::fp threshold) const {
  // calculate new accumulated probability
  const auto c = w;
  assert(std::abs(c.imag()) < EPS &&
         "Density matrix diagonal must be real-valued.");
  const auto val = prob * c.real();

  if (val < threshold) {
    return;
  }

  if (isTerminal()) {
    f(i, val);
    return;
  }

  // recursive case
  if (auto& e = p->e[0]; e.w != 0.) {
    e.traverseDiagonal(val, i, f, threshold);
  }
  if (auto& e = p->e[3]; e.w != 0.) {
    e.traverseDiagonal(val, i | (1ULL << p->v), f, threshold);
  }
}

///-----------------------------------------------------------------------------
///                      \n Explicit instantiations \n
///-----------------------------------------------------------------------------

template struct Edge<vNode>;
template struct Edge<mNode>;
template struct Edge<dNode>;

template std::complex<fp>
Edge<vNode>::getValueByIndex<vNode, true>(const std::size_t i) const;
template CVec Edge<vNode>::getVector<vNode, true>(const fp threshold) const;
template SparseCVec
Edge<vNode>::getSparseVector<vNode, true>(const fp threshold) const;
template void Edge<vNode>::printVector<vNode, true>() const;
template void Edge<vNode>::addToVector<vNode, true>(CVec& amplitudes) const;
template void
Edge<vNode>::traverseVector<vNode, true>(const std::complex<fp>& amp,
                                         const std::size_t i, AmplitudeFunc f,
                                         const fp threshold) const;

template std::complex<fp>
Edge<mNode>::getValueByIndex<mNode, true>(const std::size_t i,
                                          const std::size_t j) const;
template CMat Edge<mNode>::getMatrix<mNode, true>(const fp threshold) const;
template SparseCMat
Edge<mNode>::getSparseMatrix<mNode, true>(const fp threshold) const;
template void Edge<mNode>::printMatrix<mNode, true>() const;
template void Edge<mNode>::traverseMatrix<mNode, true>(
    const std::complex<fp>& amp, const std::size_t i, const std::size_t j,
    MatrixEntryFunc f, const fp threshold) const;

template CMat Edge<dNode>::getMatrix<dNode, true>(const fp threshold) const;
template SparseCMat
Edge<dNode>::getSparseMatrix<dNode, true>(const fp threshold) const;
template void Edge<dNode>::printMatrix<dNode, true>() const;
template SparsePVec
Edge<dNode>::getSparseProbabilityVector(const fp threshold) const;
template SparsePVecStrKeys
Edge<dNode>::getSparseProbabilityVectorStrKeys(const fp threshold) const;
template std::complex<fp>
Edge<mNode>::getValueByIndex<dNode, true>(const std::size_t i,
                                          const std::size_t j) const;
template void Edge<mNode>::traverseMatrix<dNode, true>(
    const std::complex<fp>& amp, const std::size_t i, const std::size_t j,
    MatrixEntryFunc f, const fp threshold) const;
template void Edge<dNode>::traverseDiagonal(const fp& prob, const std::size_t i,
                                            ProbabilityFunc f,
                                            const dd::fp threshold) const;

} // namespace dd

///-----------------------------------------------------------------------------
///                         \n Hash related code \n
///-----------------------------------------------------------------------------

namespace std {
template <class Node>
std::size_t
hash<dd::Edge<Node>>::operator()(const dd::Edge<Node>& e) const noexcept {
  const auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
  const auto h2 =
      dd::murmur64(static_cast<std::size_t>(std::round(e.w.real() / dd::EPS)));
  const auto h3 =
      dd::murmur64(static_cast<std::size_t>(std::round(e.w.imag() / dd::EPS)));
  auto h4 = dd::combineHash(dd::combineHash(h1, h2), h3);
  if constexpr (std::is_same_v<Node, dd::dNode>) {
    if (e.isTerminal()) {
      return h4;
    }
    assert((dd::dNode::isDensityMatrixTempFlagSet(e.p)) == false);
    const auto h5 = dd::dNode::getDensityMatrixTempFlags(e.p->flags);
    h4 = dd::combineHash(h4, h5);
  }
  return h4;
}

template struct hash<dd::Edge<dd::vNode>>;
template struct hash<dd::Edge<dd::mNode>>;
template struct hash<dd::Edge<dd::dNode>>;
} // namespace std
