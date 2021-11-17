/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_QFT_H
#define QFR_QFT_H

#include "QuantumComputation.hpp"

namespace qc {
    class QFT: public QuantumComputation {
    public:
        explicit QFT(dd::QubitCount nq, bool includeMeasurements = true, bool dynamic = false);

        std::ostream& printStatistics(std::ostream& os) const override;

        dd::QubitCount precision{};
        bool           includeMeasurements = true;
        bool           dynamic             = false;

    protected:
        void createCircuit();
    };
} // namespace qc

#endif //QFR_QFT_H
