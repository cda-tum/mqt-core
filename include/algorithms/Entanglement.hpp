/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#ifndef QFR_ENTANGLEMENT_H
#define QFR_ENTANGLEMENT_H

#include <QuantumComputation.hpp>

namespace qc {
    class Entanglement: public QuantumComputation {
    public:
        explicit Entanglement(dd::QubitCount nq);
    };
} // namespace qc

#endif //QFR_ENTANGLEMENT_H
