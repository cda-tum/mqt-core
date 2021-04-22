/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "operations/NonUnitaryOperation.hpp"

#include <utility>

namespace qc {
    // Measurement constructor
    NonUnitaryOperation::NonUnitaryOperation(const dd::QubitCount nq, std::vector<dd::Qubit> qubitRegister, std::vector<std::size_t> classicalRegister):
        qubits(std::move(qubitRegister)), classics(std::move(classicalRegister)) {
        assert(qubitRegister.size() == classicalRegister.size());
        // i-th qubit to be measured shall be measured into i-th classical register
        type    = Measure;
        nqubits = nq;
        Operation::setName();
    }
    NonUnitaryOperation::NonUnitaryOperation(dd::QubitCount nq, dd::Qubit qubit, std::size_t clbit) {
        type    = Measure;
        nqubits = nq;
        qubits.emplace_back(qubit);
        classics.emplace_back(clbit);
        Operation::setName();
    }

    // Snapshot constructor
    NonUnitaryOperation::NonUnitaryOperation(const dd::QubitCount nq, const std::vector<dd::Qubit>& qubitRegister, std::size_t n):
        NonUnitaryOperation(nq, qubitRegister, Snapshot) {
        parameter[0] = static_cast<dd::fp>(n);
    }

    // General constructor
    NonUnitaryOperation::NonUnitaryOperation(const dd::QubitCount nq, const std::vector<dd::Qubit>& qubitRegister, OpType op) {
        type    = op;
        nqubits = nq;
        targets = qubitRegister;
        Operation::setName();
    }

    std::ostream& NonUnitaryOperation::printNonUnitary(std::ostream& os, const std::vector<dd::Qubit>& q, const std::vector<std::size_t>& c) const {
        auto qubitIt   = q.cbegin();
        auto classicIt = c.cbegin();
        switch (type) {
            case Measure:
                os << name << "\t";
                for (int i = 0; i < nqubits; ++i) {
                    if (qubitIt != q.cend() && *qubitIt == i) {
                        os << "\033[34m" << static_cast<std::size_t>(*classicIt) << "\t"
                           << "\033[0m";
                        ++qubitIt;
                        ++classicIt;
                    } else {
                        os << "|\t";
                    }
                }
                break;
            case Reset:
                os << name << "\t";
                for (int i = 0; i < nqubits; ++i) {
                    if (qubitIt != q.cend() && *qubitIt == i) {
                        os << "\033[31m"
                           << "r\t"
                           << "\033[0m";
                        ++qubitIt;
                    } else {
                        os << "|\t";
                    }
                }
                break;
            case Snapshot:
                os << name << "\t";
                for (int i = 0; i < nqubits; ++i) {
                    if (qubitIt != q.cend() && *qubitIt == i) {
                        os << "\033[33m"
                           << "s\t"
                           << "\033[0m";
                        ++qubitIt;
                    } else {
                        os << "|\t";
                    }
                }
                os << "\tp: (" << q.size() << ") (" << parameter[1] << ")";
                break;
            case ShowProbabilities:
                os << name;
                break;
            case Barrier:
                os << name << "\t";
                for (int i = 0; i < nqubits; ++i) {
                    if (qubitIt != q.cend() && *qubitIt == i) {
                        os << "\033[32m"
                           << "b\t"
                           << "\033[0m";
                        ++qubitIt;
                    } else {
                        os << "|\t";
                    }
                }
                break;
            default:
                std::cerr << "Non-unitary operation with invalid type " << type << " detected. Proceed with caution!" << std::endl;
                break;
        }
        return os;
    }

    void NonUnitaryOperation::dumpOpenQASM(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg) const {
        auto classicsIt = classics.cbegin();
        switch (type) {
            case Measure:
                if (isWholeQubitRegister(qreg, qubits.front(), qubits.back()) &&
                    isWholeQubitRegister(qreg, classics.front(), classics.back())) {
                    of << "measure " << qreg[qubits.front()].first << " -> " << creg[classics.front()].first << ";" << std::endl;
                } else {
                    for (const auto& c: qubits) {
                        of << "measure " << qreg[c].second << " -> " << creg[*classicsIt].second << ";" << std::endl;
                        ++classicsIt;
                    }
                }
                break;
            case Reset:
                if (isWholeQubitRegister(qreg, targets.front(), targets.back())) {
                    of << "reset " << qreg[targets.front()].first << ";" << std::endl;
                } else {
                    for (const auto& target: targets) {
                        of << "reset " << qreg[target].second << ";" << std::endl;
                    }
                }
                break;
            case Snapshot:
                if (!targets.empty()) {
                    of << "snapshot(" << parameter[0] << ") ";

                    for (unsigned int q = 0; q < targets.size(); ++q) {
                        if (q > 0) {
                            of << ", ";
                        }
                        of << qreg[targets[q]].second;
                    }
                    of << ";" << std::endl;
                }
                break;
            case ShowProbabilities:
                of << "show_probabilities;" << std::endl;
                break;
            case Barrier:
                if (isWholeQubitRegister(qreg, targets.front(), targets.back())) {
                    of << "barrier " << qreg[targets.front()].first << ";" << std::endl;
                } else {
                    for (const auto& target: targets) {
                        of << "barrier " << qreg[target].second << ";" << std::endl;
                    }
                }
                break;
            default:
                std::cerr << "Non-unitary operation with invalid type " << type << " detected. Proceed with caution!" << std::endl;
                break;
        }
    }

    void NonUnitaryOperation::dumpQiskit(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg, const char*) const {
        switch (type) {
            case Measure:
                if (isWholeQubitRegister(qreg, qubits.front(), qubits.back()) &&
                    isWholeQubitRegister(qreg, classics.front(), classics.back())) {
                    of << "qc.measure(" << qreg[qubits.front()].first << ", " << creg[classics.front()].first << ")" << std::endl;
                } else {
                    of << "qc.measure([";
                    for (const auto& q: qubits) {
                        of << qreg[q].second << ", ";
                    }
                    of << "], [";
                    for (const auto& target: classics) {
                        of << creg[target].second << ", ";
                    }
                    of << "])" << std::endl;
                }
                break;
            case Reset:
                if (isWholeQubitRegister(qreg, targets.front(), targets.back())) {
                    of << "append(Reset(), " << qreg[targets.front()].first << ", [])" << std::endl;
                } else {
                    of << "append(Reset(), [";
                    for (const auto& target: targets) {
                        of << qreg[target].second << ", " << std::endl;
                    }
                    of << "], [])" << std::endl;
                }
                break;
            case Snapshot:
                if (!targets.empty()) {
                    of << "qc.snapshot(" << parameter[0] << ", qubits=[";
                    for (const auto& target: targets) {
                        of << qreg[target].second << ", ";
                    }
                    of << "])" << std::endl;
                }
                break;
            case ShowProbabilities:
                std::cerr << "No equivalent to show_probabilities statement in qiskit" << std::endl;
                break;
            case Barrier:
                if (isWholeQubitRegister(qreg, targets.front(), targets.back())) {
                    of << "qc.barrier(" << qreg[targets.front()].first << ")" << std::endl;
                } else {
                    of << "qc.barrier([";
                    for (const auto& target: targets) {
                        of << qreg[target].first << ", ";
                    }
                    of << "])" << std::endl;
                }
                break;
            default:
                std::cerr << "Non-unitary operation with invalid type " << type << " detected. Proceed with caution!" << std::endl;
                break;
        }
    }

    bool NonUnitaryOperation::actsOn(dd::Qubit i) const {
        if (type == Measure) {
            return std::any_of(qubits.cbegin(), qubits.cend(), [&i](const auto& q) { return q == i; });
        } else if (type == Reset) {
            return std::any_of(targets.cbegin(), targets.cend(), [&i](const auto& t) { return t == i; });
        }
        return false; // other non-unitary operations (e.g., barrier statements) may be ignored
    }

    MatrixDD NonUnitaryOperation::getDD(std::unique_ptr<dd::Package>& dd, [[maybe_unused]] const dd::Controls& controls, [[maybe_unused]] const Targets& targets) const {
        // these operations do not alter the current state
        if (type == ShowProbabilities || type == Barrier || type == Snapshot) {
            return dd->makeIdent(nqubits);
        }

        throw QFRException("DD for non-unitary operation not available!");
    }
    MatrixDD NonUnitaryOperation::getInverseDD(std::unique_ptr<dd::Package>& dd, [[maybe_unused]] const dd::Controls& controls, [[maybe_unused]] const Targets& targets) const {
        // these operations do not alter the current state
        if (type == ShowProbabilities || type == Barrier || type == Snapshot) {
            return dd->makeIdent(nqubits);
        }

        throw QFRException("Non-unitary operation is not reversible! No inverse DD is available.");
    }
} // namespace qc
