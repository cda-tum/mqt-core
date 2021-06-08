/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_QPE_HPP
#define QFR_QPE_HPP

#include <QuantumComputation.hpp>

namespace qc {
    class QPE: public QuantumComputation {
    public:
        dd::fp         lambda    = 0.;
        dd::QubitCount precision = 0;

        QPE(dd::fp lambda, dd::QubitCount precision);

        std::ostream& printStatistics(std::ostream& os) const override;
    };
} // namespace qc
#endif //QFR_QPE_HPP
