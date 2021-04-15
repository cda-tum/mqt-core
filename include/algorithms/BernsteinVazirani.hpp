/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_BV_H
#define QFR_BV_H

#include <QuantumComputation.hpp>

namespace qc {
    class BernsteinVazirani: public QuantumComputation {
    protected:
        void setup();
        void oracle();
        void postProcessing();
        void full_BernsteinVazirani();

    public:
        unsigned long hiddenInteger = 0;
        unsigned int  size          = -1;

        explicit BernsteinVazirani(unsigned long hiddenInt);

        std::ostream& printStatistics(std::ostream& os) const override;
    };
} // namespace qc
#endif //QFR_BV_H
