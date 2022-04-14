/*
* This file is part of JKQ QFR library which is released under the MIT license.
* See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
*/

#include "eccs/Q18SurfaceEcc.hpp"

Q18SurfaceEcc::Q18SurfaceEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly):
    Ecc({ID::Q18Surface, 36, 8, Q18SurfaceEcc::getName()}, qc, measureFq, decomposeMC, cliffOnly) {}

void Q18SurfaceEcc::initMappedCircuit() {
    //method is overridden because we need 2 kinds of classical measurement output registers
    qc.stripIdleQubits(true, false);
    statistics.nInputQubits         = qc.getNqubits();
    statistics.nInputClassicalBits  = (int)qc.getNcbits();
    statistics.nOutputQubits        = qc.getNqubits() * 36;
    statistics.nOutputClassicalBits = statistics.nInputClassicalBits;
    qcMapped.addQubitRegister(statistics.nOutputQubits);
    qcMapped.addClassicalRegister(statistics.nInputClassicalBits);
    qcMapped.addClassicalRegister(8, "qeccX");
    qcMapped.addClassicalRegister(8, "qeccZ");
}

void Q18SurfaceEcc::writeEncoding() {
    if (!decodingDone) {
        return;
    }
    decodingDone = false;
    measureAndCorrect();
}

void Q18SurfaceEcc::measureAndCorrect() {
    if (decodingDone) {
        return;
    }
    const int nQubits    = qc.getNqubits();
    const int clAncStart = (int)qc.getNcbits();
    for (int i = 0; i < nQubits; i++) {
        dd::Qubit   q[36];  //qubits
        dd::Control cq[36]; //qubit controls
        for (int j = 0; j < 36; j++) { q[j] = dd::Qubit(i + j * nQubits); }
        for (int j = 0; j < 36; j++) { cq[j] = dd::Control{dd::Qubit(q[j]), dd::Control::Type::pos}; }
        int ancillaIndices[] = {0, 2, 4, 7, 9, 11, 12, 14, 16, 19, 21, 23, 24, 26, 28, 31, 33, 35};
        for (int ai: ancillaIndices) {
            qcMapped.reset(q[ai]);
        }

        //initialize ancillas: Z-check
        qcMapped.x(q[0], cq[1]);
        qcMapped.x(q[0], cq[6]);

        qcMapped.x(q[2], cq[1]);
        qcMapped.x(q[2], cq[8]);
        qcMapped.x(q[2], cq[3]);

        qcMapped.x(q[4], cq[3]);
        qcMapped.x(q[4], cq[10]);
        qcMapped.x(q[4], cq[5]);

        qcMapped.x(q[12], cq[6]);
        qcMapped.x(q[12], cq[18]);
        qcMapped.x(q[12], cq[13]);

        qcMapped.x(q[16], cq[10]);
        qcMapped.x(q[16], cq[15]);
        qcMapped.x(q[16], cq[17]);
        qcMapped.x(q[16], cq[22]);

        qcMapped.x(q[24], cq[18]);
        qcMapped.x(q[24], cq[25]);
        qcMapped.x(q[24], cq[30]);

        qcMapped.x(q[26], cq[20]);
        qcMapped.x(q[26], cq[25]);
        qcMapped.x(q[26], cq[27]);
        qcMapped.x(q[26], cq[32]);

        qcMapped.x(q[28], cq[22]);
        qcMapped.x(q[28], cq[27]);
        qcMapped.x(q[28], cq[29]);
        qcMapped.x(q[28], cq[34]);

        //initialize ancillas: X-check
        int x_checks[] = {7, 9, 11, 19, 23, 31, 33, 35};
        for (int xc: x_checks) {
            qcMapped.h(q[xc]);
        }

        qcMapped.x(q[1], cq[7]);
        qcMapped.x(q[6], cq[7]);
        qcMapped.x(q[8], cq[7]);
        qcMapped.x(q[13], cq[7]);

        qcMapped.x(q[3], cq[9]);
        qcMapped.x(q[8], cq[9]);
        qcMapped.x(q[10], cq[9]);
        qcMapped.x(q[15], cq[9]);

        qcMapped.x(q[5], cq[11]);
        qcMapped.x(q[10], cq[11]);
        qcMapped.x(q[17], cq[11]);

        qcMapped.x(q[13], cq[19]);
        qcMapped.x(q[18], cq[19]);
        qcMapped.x(q[20], cq[19]);
        qcMapped.x(q[25], cq[19]);

        qcMapped.x(q[17], cq[23]);
        qcMapped.x(q[22], cq[23]);
        qcMapped.x(q[29], cq[23]);

        qcMapped.x(q[25], cq[31]);
        qcMapped.x(q[30], cq[31]);
        qcMapped.x(q[32], cq[31]);

        qcMapped.x(q[27], cq[33]);
        qcMapped.x(q[32], cq[33]);
        qcMapped.x(q[34], cq[33]);

        qcMapped.x(q[29], cq[35]);
        qcMapped.x(q[34], cq[35]);

        for (int xc: x_checks) {
            qcMapped.h(q[xc]);
        }

        //map ancillas to classical bit result

        qcMapped.measure(q[0], clAncStart);
        qcMapped.measure(q[2], clAncStart + 1);
        qcMapped.measure(q[4], clAncStart + 2);
        qcMapped.measure(q[12], clAncStart + 3);
        qcMapped.measure(q[16], clAncStart + 4);
        qcMapped.measure(q[24], clAncStart + 5);
        qcMapped.measure(q[26], clAncStart + 6);
        qcMapped.measure(q[28], clAncStart + 7);

        qcMapped.measure(q[7], clAncStart + 8);
        qcMapped.measure(q[9], clAncStart + 9);
        qcMapped.measure(q[11], clAncStart + 10);
        qcMapped.measure(q[19], clAncStart + 11);
        qcMapped.measure(q[23], clAncStart + 12);
        qcMapped.measure(q[31], clAncStart + 13);
        qcMapped.measure(q[33], clAncStart + 14);
        qcMapped.measure(q[35], clAncStart + 15);

        //logic: classical control

        //bits = 28 26 24 16 | 12 4 2 0
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b00000011, qc::X, q[1]);  //0+2
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b00000110, qc::X, q[3]);  //2+4
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b00000100, qc::X, q[5]);  //4
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b00001001, qc::X, q[6]);  //0+12
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b00000010, qc::X, q[8]);  //2
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b00010100, qc::X, q[10]); //4+16
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b00001000, qc::X, q[13]); //12
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b00010000, qc::X, q[15]); //16
        //17 not corrected -> q[15]
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b00101000, qc::X, q[18]); //12+24
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b01000000, qc::X, q[20]); //26
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b10010000, qc::X, q[22]); //16+28
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b01100000, qc::X, q[25]); //24+26
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b11000000, qc::X, q[27]); //26+28
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b10000000, qc::X, q[29]); //28
        writeClassicalControl(dd::Qubit(clAncStart), 8, 0b00100000, qc::X, q[30]); //24
        //32 not corrected -> q[20]
        //34 not corrected -> q[29]

        //bits = 35 33 31 23 | 19 11 9 7
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b00000001, qc::Z, q[1]); //7
        //3 not corrected -> q[15]
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b00000100, qc::Z, q[5]); //11
        //6 not corrected -> q[1]
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b00000011, qc::Z, q[8]);  //7+9
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b00000110, qc::Z, q[10]); //9+11
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b00001001, qc::Z, q[13]); //7+19
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b00000010, qc::Z, q[15]); //9
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b00010100, qc::Z, q[17]); //11+23
        //18 not corrected -> q[20]
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b00001000, qc::Z, q[20]); //19
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b00010000, qc::Z, q[22]); //23
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b00101000, qc::Z, q[25]); //19+31
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b01000000, qc::Z, q[27]); //33
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b10010000, qc::Z, q[29]); //23+35
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b00100000, qc::Z, q[30]); //31
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b01100000, qc::Z, q[32]); //31+33
        writeClassicalControl(dd::Qubit(clAncStart + 8), 8, 0b11000000, qc::Z, q[34]); //33+35
    }
}

void Q18SurfaceEcc::writeDecoding() {
    if (decodingDone) {
        return;
    }
    const int nQubits = qc.getNqubits();
    for (int i = 0; i < nQubits; i++) {
        qcMapped.x(dd::Qubit(i + 14 * nQubits), dd::Control{dd::Qubit(i + 8 * nQubits), dd::Control::Type::pos});
        qcMapped.x(dd::Qubit(i + 14 * nQubits), dd::Control{dd::Qubit(i + 13 * nQubits), dd::Control::Type::pos});
        qcMapped.x(dd::Qubit(i + 14 * nQubits), dd::Control{dd::Qubit(i + 15 * nQubits), dd::Control::Type::pos});
        qcMapped.x(dd::Qubit(i + 14 * nQubits), dd::Control{dd::Qubit(i + 20 * nQubits), dd::Control::Type::pos});
        qcMapped.measure(dd::Qubit(i + 14 * nQubits), i);
        qcMapped.reset(i);
        qcMapped.x(dd::Qubit(i), dd::Control{dd::Qubit(i + 14 * nQubits), dd::Control::Type::pos});
    }
    decodingDone = true;
}

void Q18SurfaceEcc::mapGate(const std::unique_ptr<qc::Operation>& gate) {
    if (decodingDone && gate->getType() != qc::Measure) {
        writeEncoding();
    }
    const int nQubits = qc.getNqubits();
    dd::Qubit i;

    //no control gate decomposition is supported
    if (gate->getNcontrols() > 2 && decomposeMultiControlledGates && gate->getType() != qc::Measure) {
        gateNotAvailableError(gate);
    } else if (gate->getNcontrols() && gate->getType() != qc::Measure) {
        //TODO support later
        gateNotAvailableError(gate);
    } else {
        qc::NonUnitaryOperation* measureGate;
        switch (gate->getType()) {
            case qc::I:
                break;
            case qc::X:
                for (std::size_t t = 0; t < gate->getNtargets(); t++) {
                    i = gate->getTargets()[t];
                    qcMapped.x(dd::Qubit(i + 15 * nQubits));
                    qcMapped.x(dd::Qubit(i + 17 * nQubits));
                }
                break;
            case qc::H:
                //apply H gate to every data qubit
                //swap circuit along '/' axis
                for (std::size_t t = 0; t < gate->getNtargets(); t++) {
                    i                              = gate->getTargets()[t];
                    std::int_fast8_t data_qubits[] = {1, 3, 5, 6, 8, 10, 13, 15, 17, 18, 20, 22, 25, 27, 29, 30, 32, 34};
                    for (std::int_fast8_t j: data_qubits) {
                        qcMapped.h(dd::Qubit(i + j * nQubits));
                    }
                    qcMapped.swap(dd::Qubit(i + 1 * nQubits), dd::Qubit(i + 29 * nQubits));
                    qcMapped.swap(dd::Qubit(i + 3 * nQubits), dd::Qubit(i + 17 * nQubits));
                    qcMapped.swap(dd::Qubit(i + 6 * nQubits), dd::Qubit(i + 34 * nQubits));
                    qcMapped.swap(dd::Qubit(i + 8 * nQubits), dd::Qubit(i + 22 * nQubits));
                    qcMapped.swap(dd::Qubit(i + 13 * nQubits), dd::Qubit(i + 27 * nQubits));
                    qcMapped.swap(dd::Qubit(i + 18 * nQubits), dd::Qubit(i + 32 * nQubits));
                    //qubits 5, 10, 15, 20, 25, 30 are along axis
                }
                break;
            case qc::Y:
                //Y = Z X
                for (std::size_t t = 0; t < gate->getNtargets(); t++) {
                    i = gate->getTargets()[t];
                    qcMapped.z(dd::Qubit(i + 18 * nQubits));
                    qcMapped.z(dd::Qubit(i + 20 * nQubits));
                    qcMapped.x(dd::Qubit(i + 15 * nQubits));
                    qcMapped.x(dd::Qubit(i + 17 * nQubits));
                }
                break;
            case qc::Z:
                for (std::size_t t = 0; t < gate->getNtargets(); t++) {
                    i = gate->getTargets()[t];
                    qcMapped.z(dd::Qubit(i + 18 * nQubits));
                    qcMapped.z(dd::Qubit(i + 20 * nQubits));
                }
                break;
            case qc::Measure:
                if (!decodingDone) {
                    measureAndCorrect();
                    writeDecoding();
                }
                measureGate = (qc::NonUnitaryOperation*)gate.get();
                for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
                    qcMapped.measure(measureGate->getTargets()[j], measureGate->getClassics()[j]);
                }
                break;
            case qc::S:
            case qc::Sdag:
            case qc::T:
            case qc::Tdag:
            case qc::V:
            case qc::Vdag:
            case qc::U3:
            case qc::U2:
            case qc::Phase:
            case qc::SX:
            case qc::SXdag:
            case qc::RX:
            case qc::RY:
            case qc::RZ:
            case qc::SWAP:
            case qc::iSWAP:
            case qc::Peres:
            case qc::Peresdag:
            case qc::Compound:
            case qc::ClassicControlled:
            default:
                gateNotAvailableError(gate);
        }
    }
}
