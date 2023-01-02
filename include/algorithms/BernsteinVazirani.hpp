/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include <QuantumComputation.hpp>
#include <bitset>

namespace qc {
    class BernsteinVazirani: public QuantumComputation {
    public:
        BitString   s        = 0;
        std::size_t bitwidth = 1;
        bool        dynamic  = false;
        std::string expected{};

        explicit BernsteinVazirani(const BitString& hiddenString, bool dyn = false);
        explicit BernsteinVazirani(std::size_t nq, bool dyn = false);
        BernsteinVazirani(const BitString& hiddenString, std::size_t nq, bool dyn = false);

        std::ostream& printStatistics(std::ostream& os) const override;

    protected:
        void createCircuit();
    };
} // namespace qc
