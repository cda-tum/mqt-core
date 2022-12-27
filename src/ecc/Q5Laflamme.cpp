/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "ecc/Q5Laflamme.hpp"

void Q5Laflamme::writeEncoding() {
    Ecc::writeEncoding();

    const auto nQubits         = qcOriginal->getNqubits();
    const auto ancStart        = static_cast<dd::Qubit>(nQubits * ecc.nRedundantQubits);
    const auto clEncode        = qcOriginal->getNcbits() + 4; //encode
    const auto controlRegister = std::make_pair(static_cast<dd::Qubit>(clEncode), static_cast<dd::QubitCount>(1));

    for (dd::Qubit i = 0; i < nQubits; i++) {
        qcMapped->reset(ancStart);
    }

    for (dd::Qubit i = 0; i < nQubits; i++) {
        qcMapped->h(ancStart);
        for (std::size_t j = 0; j < ecc.nRedundantQubits; j++) {
            qcMapped->z(static_cast<dd::Qubit>(i + j * nQubits), dd::Control{ancStart, dd::Control::Type::pos});
        }
        qcMapped->h(static_cast<dd::Qubit>(ancStart));
        qcMapped->measure(ancStart, clEncode);

        for (std::size_t j = 0; j < ecc.nRedundantQubits; j++) {
            classicalControl(controlRegister, 1, qc::OpType::X, static_cast<dd::Qubit>(i + j * nQubits));
        }
    }
    gatesWritten = true;
}

void Q5Laflamme::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    const auto nQubits    = static_cast<dd::Qubit>(qcOriginal->getNqubits());
    const auto ancStart   = static_cast<dd::Qubit>(nQubits * ecc.nRedundantQubits);
    const auto clAncStart = static_cast<dd::Qubit>(qcOriginal->getNcbits());

    for (dd::Qubit i = 0; i < nQubits; i++) {
        std::array<dd::Qubit, 5> qubits = {};
        for (std::size_t j = 0; j < qubits.size(); j++) {
            qubits[j] = static_cast<dd::Qubit>(i + j * nQubits);
        }

        //initialize ancilla qubits
        std::array<dd::Control, 4> controls;
        for (std::size_t j = 0; j < controls.size(); j++) {
            qcMapped->reset(static_cast<dd::Qubit>(ancStart + j));
            qcMapped->h(static_cast<dd::Qubit>(ancStart + j));
            controls[j] = dd::Control{static_cast<dd::Qubit>(ancStart + j), dd::Control::Type::pos};
        }

        //performes the controlled operations for ancilla qubits
        for (std::size_t c = 0; c < stabilizerMatrix.size(); c++) {
            for (std::size_t q = 0; q < stabilizerMatrix[c].size(); q++) {
                switch (stabilizerMatrix[c][q]) {
                    case qc::X: qcMapped->x(qubits[q], controls[c]); break;
                    case qc::Z: qcMapped->z(qubits[q], controls[c]); break;
                    default: break;
                }
            }
        }

        //measure ancilla qubits
        for (std::size_t j = 0; j < ecc.nCorrectingBits; j++) {
            qcMapped->h(static_cast<dd::Qubit>(ancStart + j));
            qcMapped->measure(static_cast<dd::Qubit>(ancStart + j), clAncStart + j);
        }

        const auto controlRegister = std::make_pair(static_cast<dd::Qubit>(qcOriginal->getNcbits()), static_cast<dd::QubitCount>(4));

        //perform corrections
        for (std::size_t q = 0; q < ecc.nRedundantQubits; q++) {
            for (auto op: {qc::X, qc::Y, qc::Z}) {
                std::size_t value = 0;
                for (std::size_t c = 0; c < stabilizerMatrix.size(); c++) {
                    if (!commutative(op, stabilizerMatrix[c][q])) {
                        value |= (1 << c);
                    }
                }
                classicalControl(controlRegister, value, op, qubits[q]);
            }
        }
    }
}

void Q5Laflamme::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const dd::QubitCount                      nQubits          = qcOriginal->getNqubits();
    const size_t                              clAncStart       = qcOriginal->getNcbits();
    static constexpr std::array<dd::Qubit, 8> correctionNeeded = {1, 2, 4, 7, 8, 11, 13, 14}; //values with odd amount of '1' bits

    for (std::size_t i = 0; i < nQubits; i++) {
        //#|####
        //0|1111
        //odd amount of 1's -> x[0] = 1
        //measure from index 1 (not 0) to 4, =qubit 2 to 5
        for (std::size_t j = 1; j < ecc.nRedundantQubits; j++) {
            qcMapped->measure(static_cast<dd::Qubit>(i + j * nQubits), clAncStart + j - 1);
        }
        const auto controlRegister = std::make_pair(static_cast<dd::Qubit>(clAncStart), 4);
        for (dd::Qubit const value: correctionNeeded) {
            classicalControl(controlRegister, value, qc::X, static_cast<dd::Qubit>(i));
        }
    }
    isDecoded = true;
}

void Q5Laflamme::mapGate(const qc::Operation& gate) {
    if (isDecoded && gate.getType() != qc::Measure && gate.getType() != qc::H) {
        writeEncoding();
    }
    const auto nQubits = qcOriginal->getNqubits();
    switch (gate.getType()) {
        case qc::I: break;
        case qc::X:
        case qc::Y:
        case qc::Z:
            for (std::size_t t = 0; t < gate.getNtargets(); t++) {
                auto i = gate.getTargets()[t];
                if (gate.getNcontrols() != 0U) {
                    gateNotAvailableError(gate);
                } else {
                    for (dd::Qubit j = 0; j < 5; j++) {
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), static_cast<dd::Qubit>(i + j * nQubits), gate.getType());
                    }
                }
            }
            break;
        case qc::Measure:
            if (!isDecoded) {
                measureAndCorrect();
                writeDecoding();
            }
            if (const auto* measureGate = dynamic_cast<const qc::NonUnitaryOperation*>(&gate)) {
                for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
                    qcMapped->measure(static_cast<dd::Qubit>(measureGate->getClassics()[j]), measureGate->getTargets()[j]);
                }
            } else {
                throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
            }

            break;
        default:
            gateNotAvailableError(gate);
    }
}
