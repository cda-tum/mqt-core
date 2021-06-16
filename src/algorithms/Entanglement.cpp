/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/Entanglement.hpp"

namespace qc {
    Entanglement::Entanglement(dd::QubitCount nq):
        QuantumComputation(nq) {
        name = "entanglement_" + std::to_string(nq);
        h(0);

        for (unsigned short i = 1; i < nq; i++) {
            x(i, dd::Control{0});
        }
    }
} // namespace qc
