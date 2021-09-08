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

        for (dd::QubitCount i = 0; i < precision; i++) {
            h(1);
            // normalize angle
            const auto angle = std::remainder((1U << (precision - 1 - i)) * lambda, 2.0 * dd::PI);
            phase(0, 1_pc, angle);
            for (dd::QubitCount j = 0; j < i; j++) {
                auto                           iQFT_lambda = -dd::PI / (1U << (i - j));
                std::unique_ptr<qc::Operation> op          = std::make_unique<StandardOperation>(nqubits, 1, Phase, iQFT_lambda);
                emplace_back<ClassicControlledOperation>(op, std::pair{static_cast<dd::Qubit>(j), 1U}, 1);
            }
            h(1);
            measure(1, i);
            if (i < precision - 1)
                reset(1);
        }
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
