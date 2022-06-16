/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "QuantumComputation.hpp"

namespace qc {
    class QFT: public QuantumComputation {
    public:
        explicit QFT(dd::QubitCount nq, bool includeMeasurements = true, bool dynamic = false);

        std::ostream& printStatistics(std::ostream& os) const override;

        const dd::QubitCount precision{};
        const bool           includeMeasurements;
        const bool           dynamic;

    protected:
        void createCircuit();
    };
} // namespace qc
