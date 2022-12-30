/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "QuantumComputation.hpp"
#include "algorithms/GoogleRandomCircuitSampling.hpp"
#include "dd/Operations.hpp"

namespace dd {
    using namespace qc;

    template<class Config>
    VectorDD simulate(const QuantumComputation* qc, const VectorDD& in, std::unique_ptr<dd::Package<Config>>& dd);

    template<class Config>
    std::map<std::string, std::size_t> simulate(const QuantumComputation* qc, const VectorDD& in, std::unique_ptr<dd::Package<Config>>& dd, std::size_t shots, std::size_t seed = 0U);

    template<class Config>
    void extractProbabilityVector(const QuantumComputation* qc, const VectorDD& in, dd::ProbabilityVector& probVector, std::unique_ptr<dd::Package<Config>>& dd);

    template<class Config>
    void extractProbabilityVectorRecursive(const QuantumComputation* qc, const VectorDD& currentState, decltype(qc->begin()) currentIt, std::map<std::size_t, char> measurements, dd::fp commonFactor, dd::ProbabilityVector& probVector, std::unique_ptr<dd::Package<Config>>& dd);

    template<class Config>
    VectorDD simulate(GoogleRandomCircuitSampling* qc, const VectorDD& in, std::unique_ptr<dd::Package<Config>>& dd, std::optional<std::size_t> ncycles = std::nullopt);
} // namespace dd
