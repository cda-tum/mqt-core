/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_BV_H
#define QFR_BV_H

#include <QuantumComputation.hpp>
#include <bitset>

using namespace dd::literals;

namespace qc {
    class BernsteinVazirani: public QuantumComputation {
    public:
        BitString      s        = 0;
        dd::QubitCount bitwidth = 1;
        bool           dynamic  = false;
        std::string    expected{};

        explicit BernsteinVazirani(const BitString& s, bool dynamic = false);
        explicit BernsteinVazirani(dd::QubitCount nq, bool dynamic = false);
        BernsteinVazirani(const BitString& s, dd::QubitCount nq, bool dynamic = false);

        std::ostream& printStatistics(std::ostream& os) const override;

    protected:
        void createCircuit();
    };
} // namespace qc
#endif //QFR_BV_H
