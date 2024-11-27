#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/Operations.hpp"
#include "dd/Package_fwd.hpp"
#include "ir/QuantumComputation.hpp"

#include <cstddef>
#include <map>
#include <string>

namespace dd {
using namespace qc;

template <class Config>
VectorDD simulate(const QuantumComputation* qc, const VectorDD& in,
                  Package<Config>& dd) {
  // measurements are currently not supported here
  auto permutation = qc->initialLayout;
  auto e = in;

  for (const auto& op : *qc) {
    e = applyUnitaryOperation(op.get(), e, dd, permutation);
  }

  // correct permutation if necessary
  changePermutation(e, permutation, qc->outputPermutation, dd);
  e = dd.reduceGarbage(e, qc->garbage);

  return e;
}

template <class Config>
std::map<std::string, std::size_t>
simulate(const QuantumComputation* qc, const VectorDD& in, Package<Config>& dd,
         std::size_t shots, std::size_t seed = 0U);

template <class Config>
void extractProbabilityVector(const QuantumComputation* qc, const VectorDD& in,
                              dd::SparsePVec& probVector, Package<Config>& dd);

template <class Config>
void extractProbabilityVectorRecursive(
    const QuantumComputation* qc, const VectorDD& currentState,
    decltype(qc->begin()) currentIt, Permutation& permutation,
    std::map<std::size_t, char> measurements, dd::fp commonFactor,
    dd::SparsePVec& probVector, Package<Config>& dd);
} // namespace dd
