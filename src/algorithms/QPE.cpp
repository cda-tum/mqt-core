/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "algorithms/QPE.hpp"

namespace qc {
    QPE::QPE(dd::QubitCount nq, bool exact, bool iterative):
        precision(nq), iterative(iterative) {
        if (exact) {
            // if an exact solution is wanted, generate a random n-bit number and convert it to an appropriate phase
            std::uint_least64_t max          = 1ULL << nq;
            auto                distribution = std::uniform_int_distribution<std::uint_least64_t>(0, max - 1);
            std::uint_least64_t theta        = 0;
            while (theta == 0) {
                theta = distribution(mt);
            }
            lambda = 0.;
            for (std::size_t i = 0; i < nq; ++i) {
                if (theta & (1ULL << (nq - i - 1))) {
                    lambda += 1. / static_cast<double>(1ULL << i);
                }
            }
        } else {
            // if an inexact solution is wanted, generate a random n+1-bit number (that has its last bit set) and convert it to an appropriate phase
            std::uint_least64_t max          = 1ULL << (nq + 1);
            auto                distribution = std::uniform_int_distribution<std::uint_least64_t>(0, max - 1);
            std::uint_least64_t theta        = 0;
            while (theta == 0 && (theta & 1) == 0) {
                theta = distribution(mt);
            }
            lambda = 0.;
            for (std::size_t i = 0; i <= nq; ++i) {
                if (theta & (1ULL << (nq - i))) {
                    lambda += 1. / static_cast<double>(1ULL << i);
                }
            }
        }
        createCircuit();
    }

    QPE::QPE(dd::fp lambda, dd::QubitCount precision, bool iterative):
        lambda(lambda), precision(precision), iterative(iterative) {
        createCircuit();
    }

    std::ostream& QPE::printStatistics(std::ostream& os) const {
        os << "QPE Statistics:\n";
        os << "\tn: " << static_cast<std::size_t>(nqubits + 1) << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tlambda: " << lambda << "π" << std::endl;
        os << "\tprecision: " << static_cast<std::size_t>(precision) << std::endl;
        os << "\titerative: " << iterative << std::endl;
        os << "--------------" << std::endl;
        return os;
    }

    void QPE::createCircuit() {
        addQubitRegister(1, "psi");

        if (iterative) {
            addQubitRegister(1, "q");
        } else {
            addQubitRegister(precision, "q");
        }

        addClassicalRegister(precision, "c");

        // prepare eigenvalue
        x(0);

        if (iterative) {
            for (dd::QubitCount i = 0; i < precision; i++) {
                // Hadamard
                h(1);

                // normalize angle
                const auto angle = std::remainder(static_cast<double>(1ULL << (precision - 1 - i)) * lambda, 2.0);

                // controlled phase rotation
                phase(0, 1_pc, angle * dd::PI);

                // hybrid quantum-classical inverse QFT
                for (dd::QubitCount j = 0; j < i; j++) {
                    auto iQFT_lambda = -dd::PI / static_cast<double>(1ULL << (i - j));
                    classicControlled(Phase, 1, {j, 1U}, 1U, iQFT_lambda);
                }
                h(1);

                // measure result
                measure(1, i);

                // reset qubit if not finished
                if (i < precision - 1)
                    reset(1);
            }
        } else {
            // Hadamard Layer
            for (dd::QubitCount i = 1; i <= precision; i++) {
                h(static_cast<dd::Qubit>(i));
            }

            for (dd::QubitCount i = 0; i < precision; i++) {
                // normalize angle
                const auto angle = std::remainder(static_cast<double>(1ULL << (precision - 1 - i)) * lambda, 2.0);

                // controlled phase rotation
                phase(0, dd::Control{static_cast<dd::Qubit>(1 + i)}, angle * dd::PI);

                // inverse QFT
                for (dd::QubitCount j = 1; j < 1 + i; j++) {
                    auto iQFT_lambda = -dd::PI / static_cast<double>(2ULL << (i - j));
                    if (j == i) {
                        sdag(static_cast<dd::Qubit>(1 + i), dd::Control{static_cast<dd::Qubit>(i)});
                    } else if (j == i - 1) {
                        tdag(static_cast<dd::Qubit>(1 + i), dd::Control{static_cast<dd::Qubit>(i - 1)});
                    } else {
                        phase(static_cast<dd::Qubit>(1 + i), dd::Control{static_cast<dd::Qubit>(j)}, iQFT_lambda);
                    }
                }
                h(static_cast<dd::Qubit>(1 + i));
            }

            // measure results
            for (dd::QubitCount i = 0; i < nqubits - 1; i++) {
                measure(static_cast<dd::Qubit>(i + 1), i);
            }
        }
    }
} // namespace qc
