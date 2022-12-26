/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "ecc/Q9Surface.hpp"

void Q9Surface::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    const auto nQubits    = qcOriginal->getNqubits();
    const auto ancStart   = qcOriginal->getNqubits() * ecc.nRedundantQubits;
    const auto clAncStart = qcOriginal->getNcbits();
    for (std::size_t i = 0; i < nQubits; i++) {
        std::array<dd::Qubit, 9>   qubits          = {};
        std::array<dd::Qubit, 8>   ancillaQubits   = {};
        std::array<dd::Control, 8> ancillaControls = {};
        std::array<dd::Control, 9> controlQubits   = {};
        for (std::size_t j = 0; j < 9; j++) {
            qubits.at(j) = static_cast<dd::Qubit>(i + j * nQubits);
        }
        for (std::size_t j = 0; j < 8; j++) {
            ancillaQubits.at(j) = static_cast<dd::Qubit>(ancStart + j);
        }
        if (gatesWritten) {
            for (std::size_t j = 0; j < 8; j++) {
                qcMapped->reset(ancillaQubits.at(j));
            }
        }
        for (std::size_t j = 0; j < 8; j++) {
            ancillaControls.at(j) = dd::Control{static_cast<dd::Qubit>(ancillaQubits.at(j)), dd::Control::Type::pos};
        }
        for (std::size_t j = 0; j < 9; j++) {
            controlQubits.at(j) = dd::Control{static_cast<dd::Qubit>(qubits.at(j)), dd::Control::Type::pos};
        }

        //X-type check on a0, a2, a5, a7: cx ancillaQubits->qubits
        //Z-type check on a1, a3, a4, a6: cz ancillaQubits->qubits = cx qubits->ancillaQubits, no hadamard gate
        qcMapped->h(ancillaQubits[0]);
        qcMapped->h(ancillaQubits[2]);
        qcMapped->h(ancillaQubits[5]);
        qcMapped->h(ancillaQubits[7]);

        qcMapped->x(ancillaQubits[6], controlQubits[8]);
        qcMapped->x(qubits[7], ancillaControls[5]);
        qcMapped->x(ancillaQubits[4], controlQubits[6]);
        qcMapped->x(ancillaQubits[3], controlQubits[4]);
        qcMapped->x(qubits[3], ancillaControls[2]);
        qcMapped->x(qubits[1], ancillaControls[0]);

        qcMapped->x(ancillaQubits[6], controlQubits[5]);
        qcMapped->x(ancillaQubits[4], controlQubits[3]);
        qcMapped->x(qubits[8], ancillaControls[5]);
        qcMapped->x(ancillaQubits[3], controlQubits[1]);
        qcMapped->x(qubits[4], ancillaControls[2]);
        qcMapped->x(qubits[2], ancillaControls[0]);

        qcMapped->x(qubits[6], ancillaControls[7]);
        qcMapped->x(qubits[4], ancillaControls[5]);
        qcMapped->x(ancillaQubits[4], controlQubits[7]);
        qcMapped->x(ancillaQubits[3], controlQubits[5]);
        qcMapped->x(qubits[0], ancillaControls[2]);
        qcMapped->x(ancillaQubits[1], controlQubits[3]);

        qcMapped->x(qubits[7], ancillaControls[7]);
        qcMapped->x(qubits[5], ancillaControls[5]);
        qcMapped->x(ancillaQubits[4], controlQubits[4]);
        qcMapped->x(ancillaQubits[3], controlQubits[2]);
        qcMapped->x(qubits[1], ancillaControls[2]);
        qcMapped->x(ancillaQubits[1], controlQubits[0]);

        qcMapped->h(ancillaQubits[0]);
        qcMapped->h(ancillaQubits[2]);
        qcMapped->h(ancillaQubits[5]);
        qcMapped->h(ancillaQubits[7]);

        qcMapped->measure(ancillaQubits[0], clAncStart);
        qcMapped->measure(ancillaQubits[2], clAncStart + 1);
        qcMapped->measure(ancillaQubits[5], clAncStart + 2);
        qcMapped->measure(ancillaQubits[7], clAncStart + 3);

        qcMapped->measure(ancillaQubits[1], clAncStart + 4);
        qcMapped->measure(ancillaQubits[3], clAncStart + 5);
        qcMapped->measure(ancillaQubits[4], clAncStart + 6);
        qcMapped->measure(ancillaQubits[6], clAncStart + 7);

        //logic
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 4, 1, qc::Z, qubits[2]);  //ancillaQubits[0]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 4, 2, qc::Z, qubits[3]);  //ancillaQubits[2] (or qubits[0])
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 4, 3, qc::Z, qubits[1]);  //ancillaQubits[0,2]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 4, 4, qc::Z, qubits[5]);  //ancillaQubits[5] (or qubits[8])
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 4, 6, qc::Z, qubits[4]);  //ancillaQubits[2,5]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 4, 8, qc::Z, qubits[6]);  //ancillaQubits[7]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 4, 12, qc::Z, qubits[7]); //ancillaQubits[5,7]

        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 4), 4, 1, qc::X, qubits[0]);  //ancillaQubits[1]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 4), 4, 2, qc::X, qubits[1]);  //ancillaQubits[3] (or qubits[2])
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 4), 4, 4, qc::X, qubits[7]);  //ancillaQubits[4] (or qubits[6])
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 4), 4, 5, qc::X, qubits[3]);  //ancillaQubits[1,4]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 4), 4, 6, qc::X, qubits[4]);  //ancillaQubits[3,4]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 4), 4, 8, qc::X, qubits[8]);  //ancillaQubits[6]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 4), 4, 10, qc::X, qubits[5]); //ancillaQubits[3,6]

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
        qcMapped->measure(static_cast<dd::Qubit>(i), i);
        qcMapped->measure(static_cast<dd::Qubit>(i + 4 * nQubits), i);
        qcMapped->measure(static_cast<dd::Qubit>(i + 8 * nQubits), i);
        qcMapped->x(static_cast<dd::Qubit>(i), dd::Control{static_cast<dd::Qubit>(i + 4 * nQubits), dd::Control::Type::pos});
        qcMapped->x(static_cast<dd::Qubit>(i), dd::Control{static_cast<dd::Qubit>(i + 8 * nQubits), dd::Control::Type::pos});
        qcMapped->measure(static_cast<dd::Qubit>(i), i);
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
                qcMapped->x(static_cast<dd::Qubit>(i + 2 * nQubits));
                qcMapped->x(static_cast<dd::Qubit>(i + 4 * nQubits));
                qcMapped->x(static_cast<dd::Qubit>(i + 6 * nQubits));
            }
            break;
        case qc::H:
            for (auto i: gate.getTargets()) {
                for (std::size_t j = 0; j < 9; j++) {
                    qcMapped->h(static_cast<dd::Qubit>(i + j * nQubits));
                }

                qcMapped->swap(static_cast<dd::Qubit>(i), static_cast<dd::Qubit>(i + 6 * nQubits));
                qcMapped->swap(static_cast<dd::Qubit>(i + 3 * nQubits), static_cast<dd::Qubit>(i + 7 * nQubits));
                qcMapped->swap(static_cast<dd::Qubit>(i + 2 * nQubits), static_cast<dd::Qubit>(i + 8 * nQubits));
                qcMapped->swap(static_cast<dd::Qubit>(i + nQubits), static_cast<dd::Qubit>(i + 5 * nQubits));
            }
            break;
        case qc::Y:
            //Y = Z X
            for (auto i: gate.getTargets()) {
                qcMapped->z(static_cast<dd::Qubit>(i));
                qcMapped->z(static_cast<dd::Qubit>(i + 4 * nQubits));
                qcMapped->z(static_cast<dd::Qubit>(i + 8 * nQubits));
                qcMapped->x(static_cast<dd::Qubit>(i + 2 * nQubits));
                qcMapped->x(static_cast<dd::Qubit>(i + 4 * nQubits));
                qcMapped->x(static_cast<dd::Qubit>(i + 6 * nQubits));
            }
            break;
        case qc::Z:
            for (auto i: gate.getTargets()) {
                qcMapped->z(static_cast<dd::Qubit>(i));
                qcMapped->z(static_cast<dd::Qubit>(i + 4 * nQubits));
                qcMapped->z(static_cast<dd::Qubit>(i + 8 * nQubits));
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
