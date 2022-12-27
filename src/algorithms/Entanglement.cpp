/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "algorithms/Entanglement.hpp"

namespace qc {
    Entanglement::Entanglement(const std::size_t nq):
        QuantumComputation(nq) {
        name           = "entanglement_" + std::to_string(nq);
        const auto top = static_cast<Qubit>(nq - 1);

        h(top);
        for (std::size_t i = 1; i < nq; i++) {
            x(static_cast<Qubit>(top - i), qc::Control{top});
        }
    }
} // namespace qc
