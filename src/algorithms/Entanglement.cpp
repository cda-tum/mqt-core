/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/Entanglement.hpp"

using namespace dd::literals;

namespace qc {
    Entanglement::Entanglement(dd::QubitCount nq):
        QuantumComputation(nq) {
        name           = "entanglement_" + std::to_string(nq);
        const auto top = static_cast<dd::Qubit>(nq - 1);

        h(top);
        for (dd::QubitCount i = 1; i < nq; i++) {
            x(static_cast<dd::Qubit>(top - i), dd::Control{top});
        }
    }
} // namespace qc
