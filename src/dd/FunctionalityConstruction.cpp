#include "dd/FunctionalityConstruction.hpp"

#include "dd/Package.hpp"
#include "ir/QuantumComputation.hpp"

#include <cmath>
#include <cstddef>
#include <stack>

namespace dd {
template <class Config>
MatrixDD buildFunctionality(const QuantumComputation* qc, Package<Config>& dd) {
  const auto nq = qc->getNqubits();
  if (nq == 0U) {
    return MatrixDD::one();
  }

  auto permutation = qc->initialLayout;
  auto e = dd.createInitialMatrix(qc->ancillary);

  for (const auto& op : *qc) {
    auto tmp = dd.multiply(getDD(op.get(), dd, permutation), e);

    dd.incRef(tmp);
    dd.decRef(e);
    e = tmp;

    dd.garbageCollect();
  }
  // correct permutation if necessary
  changePermutation(e, permutation, qc->outputPermutation, dd);
  e = dd.reduceAncillae(e, qc->ancillary);
  e = dd.reduceGarbage(e, qc->garbage);

  return e;
}

template <class Config>
MatrixDD buildFunctionalityRecursive(const QuantumComputation* qc,
                                     Package<Config>& dd) {
  if (qc->getNqubits() == 0U) {
    return MatrixDD::one();
  }

  auto permutation = qc->initialLayout;

  if (qc->size() == 1U) {
    auto e = getDD(qc->front().get(), dd, permutation);
    dd.incRef(e);
    return e;
  }

  std::stack<MatrixDD> s{};
  auto depth = static_cast<std::size_t>(std::ceil(std::log2(qc->size())));
  buildFunctionalityRecursive(qc, depth, 0, s, permutation, dd);
  auto e = s.top();
  s.pop();

  // correct permutation if necessary
  changePermutation(e, permutation, qc->outputPermutation, dd);
  e = dd.reduceAncillae(e, qc->ancillary);
  e = dd.reduceGarbage(e, qc->garbage);

  return e;
}

template <class Config>
bool buildFunctionalityRecursive(const QuantumComputation* qc,
                                 std::size_t depth, std::size_t opIdx,
                                 std::stack<MatrixDD>& s,
                                 Permutation& permutation,
                                 Package<Config>& dd) {
  // base case
  if (depth == 1U) {
    auto e = getDD(qc->at(opIdx).get(), dd, permutation);
    ++opIdx;
    if (opIdx == qc->size()) { // only one element was left
      s.push(e);
      dd.incRef(e);
      return false;
    }
    auto f = getDD(qc->at(opIdx).get(), dd, permutation);
    s.push(dd.multiply(f, e)); // ! reverse multiplication
    dd.incRef(s.top());
    return (opIdx != qc->size() - 1U);
  }

  // in case no operations are left after the first recursive call nothing has
  // to be done
  const size_t leftIdx =
      opIdx & ~(static_cast<std::size_t>(1U) << (depth - 1U));
  if (!buildFunctionalityRecursive(qc, depth - 1U, leftIdx, s, permutation,
                                   dd)) {
    return false;
  }

  const size_t rightIdx =
      opIdx | (static_cast<std::size_t>(1U) << (depth - 1U));
  const auto success =
      buildFunctionalityRecursive(qc, depth - 1U, rightIdx, s, permutation, dd);

  // get latest two results from stack and push their product on the stack
  auto e = s.top();
  s.pop();
  auto f = s.top();
  s.pop();
  s.push(dd.multiply(e, f)); // ordering because of stack structure

  // reference counting
  dd.decRef(e);
  dd.decRef(f);
  dd.incRef(s.top());
  dd.garbageCollect();

  return success;
}

template MatrixDD buildFunctionality(const qc::QuantumComputation* qc,
                                     Package<DDPackageConfig>& dd);
template MatrixDD
buildFunctionality(const qc::QuantumComputation* qc,
                   Package<dd::DensityMatrixSimulatorDDPackageConfig>& dd);
template MatrixDD
buildFunctionality(const qc::QuantumComputation* qc,
                   Package<dd::StochasticNoiseSimulatorDDPackageConfig>& dd);

template MatrixDD buildFunctionalityRecursive(const qc::QuantumComputation* qc,
                                              Package<DDPackageConfig>& dd);
template bool buildFunctionalityRecursive(const qc::QuantumComputation* qc,
                                          const std::size_t depth,
                                          const std::size_t opIdx,
                                          std::stack<MatrixDD>& s,
                                          qc::Permutation& permutation,
                                          Package<DDPackageConfig>& dd);
} // namespace dd
