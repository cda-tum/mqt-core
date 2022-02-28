/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_GROVER_H
#define QFR_GROVER_H

#include <QuantumComputation.hpp>
#include <bitset>
#include <functional>
#include <random>

namespace qc {
    class Grover: public QuantumComputation {
    protected:
        void setup(QuantumComputation& qc) const;

        void oracle(QuantumComputation& qc) const;

        void diffusion(QuantumComputation& qc) const;

        void full_grover(QuantumComputation& qc) const;

    public:
        std::size_t    seed        = 0;
        BitString      targetValue = 0;
        std::size_t    iterations  = 1;
        std::string    expected{};
        dd::QubitCount nDataQubits{};

        explicit Grover(dd::QubitCount nq, std::size_t seed = 0);

        MatrixDD buildFunctionality(std::unique_ptr<dd::Package>& dd) const override;
        MatrixDD buildFunctionalityRecursive(std::unique_ptr<dd::Package>& dd) const override;

        std::ostream& printStatistics(std::ostream& os) const override;
    };
} // namespace qc

#endif //QFR_GROVER_H
