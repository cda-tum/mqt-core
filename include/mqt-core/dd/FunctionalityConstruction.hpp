#pragma once

#include "dd/Operations.hpp"
#include "dd/Package_fwd.hpp"
#include "ir/Permutation.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/OpType.hpp"

#include <cstddef>
#include <memory>
#include <ostream>
#include <stack>
#include <vector>

namespace dd {
using namespace qc;

template <class Config>
MatrixDD buildFunctionality(const QuantumComputation* qc, Package<Config>& dd);

template <class Config>
MatrixDD buildFunctionalityRecursive(const QuantumComputation* qc,
                                     Package<Config>& dd);

template <class Config>
bool buildFunctionalityRecursive(const QuantumComputation* qc,
                                 std::size_t depth, std::size_t opIdx,
                                 std::stack<MatrixDD>& s,
                                 Permutation& permutation, Package<Config>& dd);

inline void dumpTensorNetwork(std::ostream& of, const QuantumComputation& qc) {
  of << "{\"tensors\": [\n";

  // initialize an index for every qubit
  auto inds = std::vector<std::size_t>(qc.getNqubits(), 0U);
  std::size_t gateIdx = 0U;
  auto dd = std::make_unique<dd::Package<>>(qc.getNqubits());
  for (const auto& op : qc) {
    const auto type = op->getType();
    if (op != qc.front() && (type != Measure && type != Barrier)) {
      of << ",\n";
    }
    dumpTensor(op.get(), of, inds, gateIdx, *dd);
  }
  of << "\n]}\n";
}

} // namespace dd
