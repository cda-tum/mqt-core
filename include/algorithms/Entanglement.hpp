/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include <QuantumComputation.hpp>

namespace qc {
    class Entanglement: public QuantumComputation {
    public:
        explicit Entanglement(std::size_t nq);
    };
} // namespace qc
