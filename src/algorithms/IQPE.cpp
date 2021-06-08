/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/IQPE.hpp"

namespace qc {
    IQPE::IQPE(dd::fp lambda, dd::QubitCount precision):
        lambda(lambda), precision(precision) {
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
