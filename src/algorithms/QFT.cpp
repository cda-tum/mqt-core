/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/QFT.hpp"

namespace qc {
    QFT::QFT(dd::QubitCount nq, bool includeMeasurements, bool dynamic):
        precision(nq), includeMeasurements(includeMeasurements), dynamic(dynamic) {
        name = "qft_" + std::to_string(static_cast<std::size_t>(nq));
        if (precision == 0) {
            return;
        }

        if (dynamic) {
            addQubitRegister(1);
        } else {
            addQubitRegister(precision);
        }
        addClassicalRegister(precision);
        createCircuit();
    }

    std::ostream& QFT::printStatistics(std::ostream& os) const {
        os << "QFT (" << static_cast<std::size_t>(precision) << ") Statistics:\n";
        os << "\tn: " << static_cast<std::size_t>(nqubits) << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tdynamic: " << dynamic << std::endl;
        os << "--------------" << std::endl;
        return os;
    }
    void QFT::createCircuit() {
        if (dynamic) {
            for (dd::QubitCount i = 0; i < precision; i++) {
                // apply classically controlled phase rotations
                for (dd::QubitCount j = 1; j <= i; ++j) {
                    const auto d = static_cast<dd::Qubit>(precision - j);
                    if (j == i) {
                        std::unique_ptr<qc::Operation> op = std::make_unique<StandardOperation>(nqubits, 0, S);
                        emplace_back<ClassicControlledOperation>(op, std::pair{static_cast<dd::Qubit>(d), 1U}, 1);
                    } else if (j == i - 1) {
                        std::unique_ptr<qc::Operation> op = std::make_unique<StandardOperation>(nqubits, 0, T);
                        emplace_back<ClassicControlledOperation>(op, std::pair{static_cast<dd::Qubit>(d), 1U}, 1);
                    } else {
                        auto                           powerOfTwo = std::pow(2.L, i - j + 1);
                        auto                           lambda     = static_cast<dd::fp>(dd::PI / powerOfTwo);
                        std::unique_ptr<qc::Operation> op         = std::make_unique<StandardOperation>(nqubits, 0, Phase, lambda);
                        emplace_back<ClassicControlledOperation>(op, std::pair{static_cast<dd::Qubit>(d), 1U}, 1);
                    }
                }

                // apply Hadamard
                h(0);

                // measure result
                measure(0, precision - 1 - i);

                // reset qubit if not finished
                if (i < precision - 1)
                    reset(0);
            }
        } else {
            // apply quantum Fourier transform
            for (dd::QubitCount i = 0; i < precision; ++i) {
                const auto q = static_cast<dd::Qubit>(i);

                // apply controlled rotations
                for (dd::QubitCount j = i; j > 0; --j) {
                    const auto d = static_cast<dd::Qubit>(q - j);
                    if (j == 1) {
                        s(d, dd::Control{q});
                    } else if (j == 2) {
                        t(d, dd::Control{q});
                    } else {
                        auto powerOfTwo = std::pow(2.L, j);
                        auto lambda     = static_cast<dd::fp>(dd::PI / powerOfTwo);
                        phase(d, dd::Control{q}, lambda);
                    }
                }

                // apply Hadamard
                h(q);
            }

            if (includeMeasurements) {
                // measure qubits in reverse order
                for (dd::QubitCount i = 0; i < precision; ++i) {
                    measure(static_cast<dd::Qubit>(i), precision - 1 - i);
                }
            } else {
                for (dd::Qubit i = 0; i < static_cast<dd::Qubit>(precision / 2); ++i) {
                    swap(i, static_cast<dd::Qubit>(precision - 1 - i));
                }
                for (dd::QubitCount i = 0; i < precision; ++i) {
                    outputPermutation[static_cast<dd::Qubit>(i)] = static_cast<dd::Qubit>(precision - 1 - i);
                }
            }
        }
    }
} // namespace qc
