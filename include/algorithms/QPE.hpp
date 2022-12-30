/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "QuantumComputation.hpp"

namespace qc {
    class QPE: public QuantumComputation {
    public:
        fp          lambda = 0.;
        std::size_t precision;
        bool        iterative;

        explicit QPE(std::size_t nq, bool exact = true, bool iterative = false);
        QPE(fp lambda, std::size_t precision, bool iterativ = false);

        std::ostream& printStatistics(std::ostream& os) const override;

    protected:
        void createCircuit();
    };
} // namespace qc
