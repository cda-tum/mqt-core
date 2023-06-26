#pragma once

#include "QuantumComputation.hpp"
#include "algorithms/GoogleRandomCircuitSampling.hpp"
#include "algorithms/Grover.hpp"
#include "dd/Package_fwd.hpp"

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

void dumpTensorNetwork(std::ostream& of, const QuantumComputation& qc);

} // namespace dd
