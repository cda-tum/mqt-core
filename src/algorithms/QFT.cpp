/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/QFT.hpp"

namespace qc {
    QFT::QFT(dd::QubitCount nq) {
        name = "qft_" + std::to_string(static_cast<std::size_t>(nq));
        addQubitRegister(nq);
        addClassicalRegister(nq);
        for (dd::QubitCount i = 0; i < nqubits; ++i) {
            outputPermutation[static_cast<dd::Qubit>(i)] = static_cast<dd::Qubit>(nqubits - 1 - i);
        }

        for (dd::QubitCount i = 0; i < nqubits; ++i) {
            emplace_back<StandardOperation>(nqubits, i, H);
            for (dd::QubitCount j = 1; j < nqubits - i; ++j) {
                auto powerOfTwo = std::pow(2.L, j);
                auto lambda     = static_cast<dd::fp>(dd::PI / powerOfTwo);
                if (j == 1) {
                    s(i, dd::Control{static_cast<dd::Qubit>(i + 1)});
                } else if (j == 2) {
                    t(i, dd::Control{static_cast<dd::Qubit>(i + 2)});
                } else {
                    phase(i, dd::Control{static_cast<dd::Qubit>(i + j)}, lambda);
                }
            }
        }

        for (dd::Qubit i = 0; i < static_cast<dd::Qubit>(nqubits / 2); ++i) {
            swap(i, static_cast<dd::Qubit>(nqubits - 1 - i));
        }
    }

    std::ostream& QFT::printStatistics(std::ostream& os) const {
        os << "QFT (" << static_cast<std::size_t>(nqubits) << ") Statistics:\n";
        os << "\tn: " << static_cast<std::size_t>(nqubits) << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "--------------" << std::endl;
        return os;
    }
} // namespace qc
