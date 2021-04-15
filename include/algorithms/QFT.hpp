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
        explicit QFT(unsigned short nq);

        std::ostream& printStatistics(std::ostream& os) const override;

        dd::Edge buildFunctionality(std::unique_ptr<dd::Package>& dd) const override;

        dd::Edge simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) const override;
    };
} // namespace qc

#endif //QFR_QFT_H
