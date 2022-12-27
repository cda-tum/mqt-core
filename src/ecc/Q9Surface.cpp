/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "ecc/Q9Surface.hpp"
namespace ecc {
    void Q9Surface::measureAndCorrect() {
        if (isDecoded) {
            return;
        }
        const auto nQubits    = qcOriginal->getNqubits();
        const auto ancStart   = qcOriginal->getNqubits() * ecc.nRedundantQubits;
        const auto clAncStart = qcOriginal->getNcbits();
        for (std::size_t i = 0; i < nQubits; i++) {
            std::array<Qubit, 9>       qubits          = {};
            std::array<Qubit, 8>       ancillaQubits   = {};
            std::array<dd::Control, 8> ancillaControls = {};
            std::array<dd::Control, 9> controlQubits   = {};
            for (std::size_t j = 0; j < 9; j++) {
                qubits.at(j) = static_cast<Qubit>(i + j * nQubits);
            }
            for (std::size_t j = 0; j < 8; j++) {
                ancillaQubits.at(j) = static_cast<Qubit>(ancStart + j);
            }
            if (gatesWritten) {
                for (std::size_t j = 0; j < 8; j++) {
                    qcMapped->reset(ancillaQubits.at(j));
                }
            }
            for (std::size_t j = 0; j < 8; j++) {
                ancillaControls.at(j) = dd::Control{static_cast<Qubit>(ancillaQubits.at(j)), dd::Control::Type::pos};
            }
            for (std::size_t j = 0; j < 9; j++) {
                controlQubits.at(j) = dd::Control{static_cast<Qubit>(qubits.at(j)), dd::Control::Type::pos};
            }

            //X-type check (z error) on a0, a2, a5, a7: cx ancillaQubits->qubits
            //Z-type check (x error) on a1, a3, a4, a6: cz ancillaQubits->qubits = cx qubits->ancillaQubits, no hadamard gate
            for (auto q: zAncillaQubits) {
                qcMapped->h(ancillaQubits[q]);
            }

            for (std::size_t q = 0; q < qubitCorrectionZ.size(); q++) {
                for (auto c: qubitCorrectionZ[q]) {
                    qcMapped->x(qubits[q], ancillaControls[c]);
                }
            }

            for (std::size_t q = 0; q < qubitCorrectionX.size(); q++) {
                for (auto c: qubitCorrectionX[q]) {
                    qcMapped->x(ancillaQubits[c], controlQubits[q]);
                }
            }

            for (std::size_t j = 0; j < zAncillaQubits.size(); j++) {
                qcMapped->h(ancillaQubits[zAncillaQubits[j]]);
                qcMapped->measure(ancillaQubits[zAncillaQubits[j]], clAncStart + j);
                qcMapped->measure(ancillaQubits[xAncillaQubits[j]], clAncStart + 4 + j);
            }

            //correction
            auto controlRegister = std::make_pair(static_cast<Qubit>(clAncStart), ancillaWidth);
            for (std::size_t q = 0; q < qubitCorrectionZ.size(); q++) {
                if (uncorrectedZQubits.count(static_cast<Qubit>(q)) == 0) {
                    std::size_t mask = 0;
                    for (std::size_t c = 0; c < zAncillaQubits.size(); c++) {
                        if (qubitCorrectionZ[q].count(zAncillaQubits[c]) > 0) {
                            mask |= (1 << c);
                        }
                    }
                    classicalControl(controlRegister, mask, qc::Z, qubits[q]);
                }
            }
            controlRegister = std::make_pair(static_cast<Qubit>(clAncStart + ancillaWidth), ancillaWidth);
            for (std::size_t q = 0; q < qubitCorrectionX.size(); q++) {
                if (uncorrectedXQubits.count(static_cast<Qubit>(q)) == 0) {
                    std::size_t mask = 0;
                    for (std::size_t c = 0; c < xAncillaQubits.size(); c++) {
                        if (qubitCorrectionX[q].count(xAncillaQubits[c]) > 0) {
                            mask |= (1 << c);
                        }
                    }
                    classicalControl(controlRegister, mask, qc::X, qubits[q]);
                }
            }

            gatesWritten = true;
        }
    }

    void Q9Surface::writeDecoding() {
        if (isDecoded) {
            return;
        }
        const auto nQubits = qcOriginal->getNqubits();
        for (std::size_t i = 0; i < nQubits; i++) {
            //measure 0, 4, 8. state = m0*m4*m8
            qcMapped->measure(static_cast<Qubit>(i), i);
            qcMapped->measure(static_cast<Qubit>(i + 4 * nQubits), i);
            qcMapped->measure(static_cast<Qubit>(i + 8 * nQubits), i);
            qcMapped->x(static_cast<Qubit>(i), dd::Control{static_cast<Qubit>(i + 4 * nQubits), dd::Control::Type::pos});
            qcMapped->x(static_cast<Qubit>(i), dd::Control{static_cast<Qubit>(i + 8 * nQubits), dd::Control::Type::pos});
            qcMapped->measure(static_cast<Qubit>(i), i);
        }
        isDecoded = true;
    }

    void Q9Surface::mapGate(const qc::Operation& gate) {
        if (isDecoded && gate.getType() != qc::Measure) {
            writeEncoding();
        }
        const auto nQubits = qcOriginal->getNqubits();

        if ((gate.getNcontrols() != 0U) && gate.getType() != qc::Measure) {
            gateNotAvailableError(gate);
        }

        switch (gate.getType()) {
            case qc::I:
                break;
            case qc::X:
                for (auto i: gate.getTargets()) {
                    for (auto j: logicalX) {
                        qcMapped->x(static_cast<Qubit>(i + j * nQubits));
                    }
                }
                break;
            case qc::H:
                for (auto i: gate.getTargets()) {
                    for (std::size_t j = 0; j < 9; j++) {
                        qcMapped->h(static_cast<Qubit>(i + j * nQubits));
                    }
                    for (auto pair: swapIndices) {
                        qcMapped->swap(static_cast<Qubit>(i + pair.first * nQubits), static_cast<Qubit>(i + pair.second * nQubits));
                    }
                }
                break;
            case qc::Y:
                //Y = Z X
                for (auto i: gate.getTargets()) {
                    for (auto j: logicalZ) {
                        qcMapped->z(static_cast<Qubit>(i + j * nQubits));
                    }
                    for (auto j: logicalX) {
                        qcMapped->x(static_cast<Qubit>(i + j * nQubits));
                    }
                }
                break;
            case qc::Z:
                for (auto i: gate.getTargets()) {
                    for (auto j: logicalZ) {
                        qcMapped->z(static_cast<Qubit>(i + j * nQubits));
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
                        qcMapped->measure(measureGate->getTargets().at(j), measureGate->getClassics().at(j));
                    }
                } else {
                    throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
                }
                break;
            default:
                gateNotAvailableError(gate);
        }
    }
} // namespace ecc
