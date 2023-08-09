#pragma once

#include "QuantumComputation.hpp"
#include "algorithms/GoogleRandomCircuitSampling.hpp"
#include "algorithms/Grover.hpp"
#include "dd/Operations.hpp"

namespace dd {
using namespace qc;

template <class Config>
MatrixDD buildFunctionality(const QuantumComputation* qc,
                            std::unique_ptr<dd::Package<Config>>& dd);

template <class Config>
MatrixDD buildFunctionalityRecursive(const QuantumComputation* qc,
                                     std::unique_ptr<dd::Package<Config>>& dd);

template <class Config>
bool buildFunctionalityRecursive(const QuantumComputation* qc,
                                 std::size_t depth, std::size_t opIdx,
                                 std::stack<MatrixDD>& s,
                                 Permutation& permutation,
                                 std::unique_ptr<dd::Package<Config>>& dd);

template <class Config>
MatrixDD buildFunctionality(const qc::Grover* qc,
                            std::unique_ptr<dd::Package<Config>>& dd);

template <class Config>
MatrixDD buildFunctionalityRecursive(const qc::Grover* qc,
                                     std::unique_ptr<dd::Package<Config>>& dd);

template <class DDPackage>
MatrixDD buildFunctionality(GoogleRandomCircuitSampling* qc,
                            std::unique_ptr<DDPackage>& dd,
                            std::optional<std::size_t> ncycles = std::nullopt);

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
    dumpTensor(op.get(), of, inds, gateIdx, dd);
  }
  of << "\n]}\n";
}

} // namespace dd
