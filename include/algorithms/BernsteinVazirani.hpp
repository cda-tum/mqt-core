/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

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
