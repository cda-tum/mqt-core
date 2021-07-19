/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/QPE.hpp"

namespace qc {
    QPE::QPE(dd::fp lambda, dd::QubitCount precision):
        lambda(lambda), precision(precision) {
        addQubitRegister(1, "psi");
        addQubitRegister(precision, "q");
        addClassicalRegister(precision, "c");

        //Hadamard Layer
        for (dd::QubitCount i = 1; i <= precision; i++) {
            h(static_cast<dd::Qubit>(i));
        }
        //prepare eigenvalue
        x(0);

        //Controlled Phase Rotation
        for (dd::QubitCount i = 0; i < precision; i++) {
            phase(0, dd::Control{static_cast<dd::Qubit>(i + 1)}, ((1U << i) * lambda));
        }

        //Inverse QFT
        for (dd::Qubit i = 1; i <= static_cast<dd::Qubit>((nqubits - 1) / 2); ++i) {
            swap(i, static_cast<dd::Qubit>(nqubits - i));
        }
        for (dd::QubitCount i = 1; i <= precision; ++i) {
            for (dd::QubitCount j = 1; j < i; j++) {
                auto iQFT_lambda = -dd::PI / (1U << j);
                if (j == 1) {
                    sdag(static_cast<dd::Qubit>(i - 1), dd::Control{static_cast<dd::Qubit>(i)});
                } else if (j == 2) {
                    tdag(static_cast<dd::Qubit>(i - 2), dd::Control{static_cast<dd::Qubit>(i)});
                } else {
                    phase(static_cast<dd::Qubit>(i - j), dd::Control{static_cast<dd::Qubit>(i)}, iQFT_lambda);
                }
            }
            h(static_cast<dd::Qubit>(i));
        }

        //Measure Results
        for (dd::QubitCount i = 0; i < nqubits - 1; i++) {
            measure(static_cast<dd::Qubit>(i + 1), i);
        }
    }

    std::ostream& QPE::printStatistics(std::ostream& os) const {
        os << "QPE Statistics:\n";
        os << "\tn: " << static_cast<std::size_t>(nqubits + 1) << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tlambda: " << lambda << std::endl;
        os << "\tprecision: " << static_cast<std::size_t>(precision) << std::endl;
        os << "--------------" << std::endl;
        return os;
    }
} // namespace qc
