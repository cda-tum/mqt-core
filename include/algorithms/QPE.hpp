/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "QuantumComputation.hpp"

using namespace dd::literals;

namespace qc {
    class QPE: public QuantumComputation {
    public:
        dd::fp               lambda = 0.;
        const dd::QubitCount precision;
        const bool           iterative;

        explicit QPE(dd::QubitCount nq, bool exact = true, bool iterative = false);
        QPE(dd::fp lambda, dd::QubitCount precision, bool iterativ = false);

        std::ostream& printStatistics(std::ostream& os) const override;

    protected:
        void createCircuit();
    };
} // namespace qc
