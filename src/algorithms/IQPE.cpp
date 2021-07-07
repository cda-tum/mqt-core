/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/IQPE.hpp"

namespace qc {
    IQPE::IQPE(dd::fp lambda, dd::QubitCount precision):
        lambda(lambda), precision(precision) {
        addQubitRegister(1, "psi");
        addQubitRegister(1, "q");
        addClassicalRegister(precision, "c");

        x(static_cast<dd::Qubit>(0));

        for(dd::QubitCount i = 0; i < precision; i++){
            h(static_cast<dd::Qubit>(1));
            phase(static_cast<dd::Qubit>(0), dd::Control{static_cast<dd::Qubit>(1)}, (1U << (precision-1-i))*lambda);
            for(dd::QubitCount j = 0; j < i; j++){
                auto iQFT_lambda = -dd::PI / (1U << (i-j));
                auto op = std::make_unique<StandardOperation>(nqubits, static_cast<dd::Qubit>(1), Phase, iQFT_lambda);
                //emplace_back<ClassicControlledOperation>(op, (i-1), 1);
            }
            h(static_cast<dd::Qubit>(1));
            measure(static_cast<dd::Qubit>(1), i);
            reset(static_cast<dd::Qubit>(1));
        }


        /// TODO: Construct Iterative Quantum Phase Estimation circuit for U=p(theta) with the specified precision
    }

    std::ostream& IQPE::printStatistics(std::ostream& os) const {
        os << "IQPE Statistics:\n";
        os << "\tn: " << static_cast<std::size_t>(nqubits + 1) << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tlambda: " << lambda << std::endl;
        os << "\tprecision: " << static_cast<std::size_t>(precision) << std::endl;
        os << "--------------" << std::endl;
        return os;
    }
} // namespace qc
