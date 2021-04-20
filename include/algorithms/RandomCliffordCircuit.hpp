/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_RANDOMCLIFFORDCIRCUIT_HPP
#define QFR_RANDOMCLIFFORDCIRCUIT_HPP

#include <QuantumComputation.hpp>
#include <functional>
#include <random>

namespace qc {
    class RandomCliffordCircuit: public QuantumComputation {
    protected:
        std::function<std::uint_fast16_t()> cliffordGenerator;

        void append1QClifford(std::uint_fast16_t idx, dd::Qubit target);
        void append2QClifford(std::uint_fast16_t, dd::Qubit control, dd::Qubit target);

    public:
        std::size_t depth = 1;
        std::size_t seed  = 0;

        explicit RandomCliffordCircuit(dd::QubitCount nq, std::size_t depth = 1, std::size_t seed = 0);

        std::ostream& printStatistics(std::ostream& os) const override;
    };
} // namespace qc

#endif //QFR_RANDOMCLIFFORDCIRCUIT_HPP
