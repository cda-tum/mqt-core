/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "ecc/Q18Surface.hpp"

void Q18Surface::initMappedCircuit() {
    //method is overridden because we need 2 kinds of classical measurement output registers
    qcOriginal->stripIdleQubits(true, false);
    qcMapped->addQubitRegister(getNOutputQubits(qcOriginal->getNqubits()));
    auto cRegs = qcOriginal->getCregs();
    for (auto const& [regName, regBits]: cRegs) {
        qcMapped->addClassicalRegister(regBits.second, regName.c_str());
    }
    qcMapped->addClassicalRegister(8, "qeccX");
    qcMapped->addClassicalRegister(8, "qeccZ");
}

void Q18Surface::writeEncoding() {
    if (!isDecoded) {
        return;
    }
    isDecoded = false;
    measureAndCorrect();
}

void Q18Surface::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    const auto nQubits    = qcOriginal->getNqubits();
    const auto clAncStart = qcOriginal->getNcbits();
    for (dd::Qubit i = 0; i < nQubits; i++) {
        std::array<dd::Qubit, 36>   qubits        = {};
        std::array<dd::Control, 36> controlQubits = {};
        for (std::size_t j = 0; j < 36; j++) {
            qubits.at(j) = static_cast<dd::Qubit>(i + j * nQubits);
        }
        for (std::size_t j = 0; j < 36; j++) {
            controlQubits.at(j) = dd::Control{static_cast<dd::Qubit>(qubits.at(j)), dd::Control::Type::pos};
        }

        if (gatesWritten) {
            for (dd::Qubit const ai: ancillaIndices) {
                qcMapped->reset(qubits.at(ai));
            }
        }

        //initialize ancillas: Z-check
        qcMapped->x(qubits[0], controlQubits[1]);
        qcMapped->x(qubits[0], controlQubits[6]);

        qcMapped->x(qubits[2], controlQubits[1]);
        qcMapped->x(qubits[2], controlQubits[8]);
        qcMapped->x(qubits[2], controlQubits[3]);

        qcMapped->x(qubits[4], controlQubits[3]);
        qcMapped->x(qubits[4], controlQubits[10]);
        qcMapped->x(qubits[4], controlQubits[5]);

        qcMapped->x(qubits[12], controlQubits[6]);
        qcMapped->x(qubits[12], controlQubits[18]);
        qcMapped->x(qubits[12], controlQubits[13]);

        qcMapped->x(qubits[16], controlQubits[10]);
        qcMapped->x(qubits[16], controlQubits[15]);
        qcMapped->x(qubits[16], controlQubits[17]);
        qcMapped->x(qubits[16], controlQubits[22]);

        qcMapped->x(qubits[24], controlQubits[18]);
        qcMapped->x(qubits[24], controlQubits[25]);
        qcMapped->x(qubits[24], controlQubits[30]);

        qcMapped->x(qubits[26], controlQubits[20]);
        qcMapped->x(qubits[26], controlQubits[25]);
        qcMapped->x(qubits[26], controlQubits[27]);
        qcMapped->x(qubits[26], controlQubits[32]);

        qcMapped->x(qubits[28], controlQubits[22]);
        qcMapped->x(qubits[28], controlQubits[27]);
        qcMapped->x(qubits[28], controlQubits[29]);
        qcMapped->x(qubits[28], controlQubits[34]);

        //initialize ancillas: X-check
        static constexpr std::array<std::size_t, 8> xChecks = {7, 9, 11, 19, 23, 31, 33, 35};
        for (std::size_t const xc: xChecks) {
            qcMapped->h(qubits.at(xc));
        }

        qcMapped->x(qubits[1], controlQubits[7]);
        qcMapped->x(qubits[6], controlQubits[7]);
        qcMapped->x(qubits[8], controlQubits[7]);
        qcMapped->x(qubits[13], controlQubits[7]);

        qcMapped->x(qubits[3], controlQubits[9]);
        qcMapped->x(qubits[8], controlQubits[9]);
        qcMapped->x(qubits[10], controlQubits[9]);
        qcMapped->x(qubits[15], controlQubits[9]);

        qcMapped->x(qubits[5], controlQubits[11]);
        qcMapped->x(qubits[10], controlQubits[11]);
        qcMapped->x(qubits[17], controlQubits[11]);

        qcMapped->x(qubits[13], controlQubits[19]);
        qcMapped->x(qubits[18], controlQubits[19]);
        qcMapped->x(qubits[20], controlQubits[19]);
        qcMapped->x(qubits[25], controlQubits[19]);

        qcMapped->x(qubits[17], controlQubits[23]);
        qcMapped->x(qubits[22], controlQubits[23]);
        qcMapped->x(qubits[29], controlQubits[23]);

        qcMapped->x(qubits[25], controlQubits[31]);
        qcMapped->x(qubits[30], controlQubits[31]);
        qcMapped->x(qubits[32], controlQubits[31]);

        qcMapped->x(qubits[27], controlQubits[33]);
        qcMapped->x(qubits[32], controlQubits[33]);
        qcMapped->x(qubits[34], controlQubits[33]);

        qcMapped->x(qubits[29], controlQubits[35]);
        qcMapped->x(qubits[34], controlQubits[35]);

        for (std::size_t const xc: xChecks) {
            qcMapped->h(qubits.at(xc));
        }

        //map ancillas to classical bit result

        qcMapped->measure(qubits[0], clAncStart);
        qcMapped->measure(qubits[2], clAncStart + 1);
        qcMapped->measure(qubits[4], clAncStart + 2);
        qcMapped->measure(qubits[12], clAncStart + 3);
        qcMapped->measure(qubits[16], clAncStart + 4);
        qcMapped->measure(qubits[24], clAncStart + 5);
        qcMapped->measure(qubits[26], clAncStart + 6);
        qcMapped->measure(qubits[28], clAncStart + 7);

        qcMapped->measure(qubits[7], clAncStart + 8);
        qcMapped->measure(qubits[9], clAncStart + 9);
        qcMapped->measure(qubits[11], clAncStart + 10);
        qcMapped->measure(qubits[19], clAncStart + 11);
        qcMapped->measure(qubits[23], clAncStart + 12);
        qcMapped->measure(qubits[31], clAncStart + 13);
        qcMapped->measure(qubits[33], clAncStart + 14);
        qcMapped->measure(qubits[35], clAncStart + 15);

        //logic: classical control

        //bits = 28 26 24 16 | 12 4 2 0
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b00000011, qc::X, qubits[1]);  //0+2
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b00000110, qc::X, qubits[3]);  //2+4
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b00000100, qc::X, qubits[5]);  //4
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b00001001, qc::X, qubits[6]);  //0+12
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b00000010, qc::X, qubits[8]);  //2
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b00010100, qc::X, qubits[10]); //4+16
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b00001000, qc::X, qubits[13]); //12
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b00010000, qc::X, qubits[15]); //16
        //17 not corrected -> qubits[15]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b00101000, qc::X, qubits[18]); //12+24
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b01000000, qc::X, qubits[20]); //26
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b10010000, qc::X, qubits[22]); //16+28
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b01100000, qc::X, qubits[25]); //24+26
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b11000000, qc::X, qubits[27]); //26+28
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b10000000, qc::X, qubits[29]); //28
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 8, 0b00100000, qc::X, qubits[30]); //24
        //32 not corrected -> qubits[20]
        //34 not corrected -> qubits[29]

        //bits = 35 33 31 23 | 19 11 9 7
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b00000001, qc::Z, qubits[1]); //7
        //3 not corrected -> qubits[15]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b00000100, qc::Z, qubits[5]); //11
        //6 not corrected -> qubits[1]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b00000011, qc::Z, qubits[8]);  //7+9
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b00000110, qc::Z, qubits[10]); //9+11
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b00001001, qc::Z, qubits[13]); //7+19
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b00000010, qc::Z, qubits[15]); //9
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b00010100, qc::Z, qubits[17]); //11+23
        //18 not corrected -> qubits[20]
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b00001000, qc::Z, qubits[20]); //19
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b00010000, qc::Z, qubits[22]); //23
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b00101000, qc::Z, qubits[25]); //19+31
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b01000000, qc::Z, qubits[27]); //33
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b10010000, qc::Z, qubits[29]); //23+35
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b00100000, qc::Z, qubits[30]); //31
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b01100000, qc::Z, qubits[32]); //31+33
        writeClassicalControl(static_cast<dd::Qubit>(clAncStart + 8), 8, 0b11000000, qc::Z, qubits[34]); //33+35

        gatesWritten = true;
    }
}

void Q18Surface::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const auto nQubits = qcOriginal->getNqubits();
    for (dd::Qubit i = 0; i < nQubits; i++) {
        qcMapped->x(static_cast<dd::Qubit>(i + 14 * nQubits), dd::Control{static_cast<dd::Qubit>(i + 8 * nQubits), dd::Control::Type::pos});
        qcMapped->x(static_cast<dd::Qubit>(i + 14 * nQubits), dd::Control{static_cast<dd::Qubit>(i + 13 * nQubits), dd::Control::Type::pos});
        qcMapped->x(static_cast<dd::Qubit>(i + 14 * nQubits), dd::Control{static_cast<dd::Qubit>(i + 15 * nQubits), dd::Control::Type::pos});
        qcMapped->x(static_cast<dd::Qubit>(i + 14 * nQubits), dd::Control{static_cast<dd::Qubit>(i + 20 * nQubits), dd::Control::Type::pos});
        qcMapped->measure(static_cast<dd::Qubit>(i + 14 * nQubits), i);
        qcMapped->reset(static_cast<dd::Qubit>(i));
        qcMapped->x(static_cast<dd::Qubit>(i), dd::Control{static_cast<dd::Qubit>(i + 14 * nQubits), dd::Control::Type::pos});
    }
    isDecoded = true;
}

void Q18Surface::mapGate(const qc::Operation& gate) {
    if (isDecoded && gate.getType() != qc::Measure) {
        writeEncoding();
    }
    const auto nQubits = qcOriginal->getNqubits();

    //no control gate decomposition is supported
    if ((gate.getNcontrols() != 0U) && gate.getType() != qc::Measure) {
        //multi-qubit gates are currently not supported
        gateNotAvailableError(gate);
    } else {
        switch (gate.getType()) {
            case qc::I:
                break;
            case qc::X:
                for (auto i: gate.getTargets()) {
                    qcMapped->x(static_cast<dd::Qubit>(i + 15 * nQubits));
                    qcMapped->x(static_cast<dd::Qubit>(i + 17 * nQubits));
                }
                break;
            case qc::H:
                //apply H gate to every data qubit
                //swap circuit along '/' axis
                for (auto i: gate.getTargets()) {
                    for (const auto j: dataQubits) {
                        qcMapped->h(static_cast<dd::Qubit>(i + j * nQubits));
                    }
                    qcMapped->swap(static_cast<dd::Qubit>(i + 1 * nQubits), static_cast<dd::Qubit>(i + 29 * nQubits));
                    qcMapped->swap(static_cast<dd::Qubit>(i + 3 * nQubits), static_cast<dd::Qubit>(i + 17 * nQubits));
                    qcMapped->swap(static_cast<dd::Qubit>(i + 6 * nQubits), static_cast<dd::Qubit>(i + 34 * nQubits));
                    qcMapped->swap(static_cast<dd::Qubit>(i + 8 * nQubits), static_cast<dd::Qubit>(i + 22 * nQubits));
                    qcMapped->swap(static_cast<dd::Qubit>(i + 13 * nQubits), static_cast<dd::Qubit>(i + 27 * nQubits));
                    qcMapped->swap(static_cast<dd::Qubit>(i + 18 * nQubits), static_cast<dd::Qubit>(i + 32 * nQubits));
                    //qubits 5, 10, 15, 20, 25, 30 are along axis
                }
                break;
            case qc::Y:
                //Y = Z X
                for (auto i: gate.getTargets()) {
                    qcMapped->z(static_cast<dd::Qubit>(i + 18 * nQubits));
                    qcMapped->z(static_cast<dd::Qubit>(i + 20 * nQubits));
                    qcMapped->x(static_cast<dd::Qubit>(i + 15 * nQubits));
                    qcMapped->x(static_cast<dd::Qubit>(i + 17 * nQubits));
                }
                break;
            case qc::Z:
                for (auto i: gate.getTargets()) {
                    qcMapped->z(static_cast<dd::Qubit>(i + 18 * nQubits));
                    qcMapped->z(static_cast<dd::Qubit>(i + 20 * nQubits));
                }
                break;
            case qc::Measure:
                if (!isDecoded) {
                    measureAndCorrect();
                    writeDecoding();
                }
                if (const auto *measureGate = dynamic_cast<const qc::NonUnitaryOperation*>(&gate)) {
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
}
