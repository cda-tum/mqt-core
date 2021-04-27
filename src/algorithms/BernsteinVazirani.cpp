/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/BernsteinVazirani.hpp"

// BernsteinVazirani without Entanglement
namespace qc {
    /***
	 * Private Methods
	 ***/
    void BernsteinVazirani::setup() {
        for (dd::QubitCount i = 0; i < nqubits; ++i)
            emplace_back<StandardOperation>(nqubits, i, H);
    }

    void BernsteinVazirani::oracle() {
        for (dd::QubitCount i = 0; i < nqubits; ++i) {
            if (((hiddenInteger >> i) & 1) == 1) {
                emplace_back<StandardOperation>(nqubits, i, Z);
            }
        }
    }

    void BernsteinVazirani::postProcessing() {
        for (dd::QubitCount i = 0; i < nqubits; ++i)
            emplace_back<StandardOperation>(nqubits, i, H);
    }

    void BernsteinVazirani::full_BernsteinVazirani() {
        // Generate circuit
        setup();
        oracle();
        postProcessing();
    }

    /***
	 * Public Methods
	 ***/
    BernsteinVazirani::BernsteinVazirani(std::size_t hiddenInteger):
        hiddenInteger(hiddenInteger) {
        name = "bv_" + std::to_string(hiddenInteger);

        if (hiddenInteger > 0) {
            size = static_cast<dd::QubitCount>(std::ceil(std::log2(hiddenInteger)));
        } else {
            size = 1;
        }

        // Set nr of Qubits/ClassicalBits
        addQubitRegister(size);
        addClassicalRegister(size);

        // Circuit
        full_BernsteinVazirani();
    }

    std::ostream& BernsteinVazirani::printStatistics(std::ostream& os) const {
        os << "BernsteinVazirani (" << static_cast<std::size_t>(nqubits) << ") Statistics:\n";
        os << "\tn: " << static_cast<std::size_t>(nqubits + 1) << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tHiddenInteger: " << hiddenInteger << std::endl;
        os << "--------------" << std::endl;
        return os;
    }
} // namespace qc
