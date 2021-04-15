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
        std::function<unsigned short()> cliffordGenerator;

        void append1QClifford(unsigned int idx, unsigned short target);
        void append2QClifford(unsigned int idx, unsigned short control, unsigned short target);

    public:
        unsigned int depth = 1;
        unsigned int seed  = 0;

        explicit RandomCliffordCircuit(unsigned short nq, unsigned int depth = 1, unsigned int seed = 0);

        std::ostream& printStatistics(std::ostream& os) const override;
    };
} // namespace qc

#endif //QFR_RANDOMCLIFFORDCIRCUIT_HPP
