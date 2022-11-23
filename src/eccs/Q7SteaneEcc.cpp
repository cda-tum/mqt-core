/*
* This file is part of JKQ QFR library which is released under the MIT license.
* See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
*/

#include "eccs/Q7SteaneEcc.hpp"

//7 data qubits, 3 for measuring
Q7SteaneEcc::Q7SteaneEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly):
    Ecc(
            {ID::Q7Steane, 7, 3, Q7SteaneEcc::getName()}, qc, measureFq, decomposeMC, cliffOnly) {}

void Q7SteaneEcc::initMappedCircuit() {
    //method is overridden because we need 2 kinds of classical measurement output registers
    qc.stripIdleQubits(true, false);
    statistics.nInputQubits         = qc.getNqubits();
    statistics.nInputClassicalBits  = (int)qc.getNcbits();
    statistics.nOutputQubits        = qc.getNqubits() * ecc.nRedundantQubits + ecc.nCorrectingBits;
    statistics.nOutputClassicalBits = statistics.nInputClassicalBits + ecc.nCorrectingBits + 7;
    qcMapped.addQubitRegister(statistics.nOutputQubits);
    //    qcMapped.addClassicalRegister(statistics.nInputClassicalBits);
    auto cRegs = qc.getCregs();
    for (auto const& [regName, regBits]: cRegs) {
        qcMapped.addClassicalRegister(regBits.second, regName);
    }
    qcMapped.addClassicalRegister(3, "qecc");
}

void Q7SteaneEcc::writeEncoding() {
    if (!isDecoded) {
        return;
    }
    isDecoded         = false;
    const int nQubits = qc.getNqubits();
    //reset data qubits if necessary
    if (gatesWritten) {
        for (int i = 0; i < nQubits; i++) {
            for (int j = 1; j < 7; j++) {
                qcMapped.reset(dd::Qubit(i + j * nQubits));
            }
        }
    }
    measureAndCorrectSingle(true);
}

void Q7SteaneEcc::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    measureAndCorrectSingle(true);
    measureAndCorrectSingle(false);
}

void Q7SteaneEcc::measureAndCorrectSingle(bool xSyndrome) {
    const int nQubits    = qc.getNqubits();
    const int ancStart   = nQubits * ecc.nRedundantQubits;
    const int clAncStart = static_cast<int>(qc.getNcbits());
    if (gatesWritten) {
        for (int i = 0; i < nQubits; i++) {
            qcMapped.reset(static_cast<dd::Qubit>(ancStart));
            qcMapped.reset(static_cast<dd::Qubit>(ancStart + 1));
            qcMapped.reset(static_cast<dd::Qubit>(ancStart + 2));
        }
    }
    for (int i = 0; i < nQubits; i++) {
        qcMapped.h(static_cast<dd::Qubit>(ancStart));
        qcMapped.h(static_cast<dd::Qubit>(ancStart + 1));
        qcMapped.h(static_cast<dd::Qubit>(ancStart + 2));

        auto c0 = dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos};
        auto c1 = dd::Control{dd::Qubit(ancStart + 1), dd::Control::Type::pos};
        auto c2 = dd::Control{dd::Qubit(ancStart + 2), dd::Control::Type::pos};

        void (*writeXZ)(dd::Qubit, dd::Control, qc::QuantumComputation*, bool) = xSyndrome ? writeXstatic : writeZstatic;

        //K1: UIUIUIU
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 0), c0, &qcMapped, cliffordGatesOnly);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 2), c0, &qcMapped, cliffordGatesOnly);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 4), c0, &qcMapped, cliffordGatesOnly);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 6), c0, &qcMapped, cliffordGatesOnly);

        //K2: IUUIIUU
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 1), c1, &qcMapped, cliffordGatesOnly);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 2), c1, &qcMapped, cliffordGatesOnly);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 5), c1, &qcMapped, cliffordGatesOnly);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 6), c1, &qcMapped, cliffordGatesOnly);

        //K3: IIIUUUU
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 3), c2, &qcMapped, cliffordGatesOnly);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 4), c2, &qcMapped, cliffordGatesOnly);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 5), c2, &qcMapped, cliffordGatesOnly);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 6), c2, &qcMapped, cliffordGatesOnly);

        qcMapped.h(static_cast<dd::Qubit>(ancStart));
        qcMapped.h(static_cast<dd::Qubit>(ancStart + 1));
        qcMapped.h(static_cast<dd::Qubit>(ancStart + 2));

        qcMapped.measure(static_cast<dd::Qubit>(ancStart), clAncStart);
        qcMapped.measure(static_cast<dd::Qubit>(ancStart + 1), clAncStart + 1);
        qcMapped.measure(static_cast<dd::Qubit>(ancStart + 2), clAncStart + 2);

        //correct Z_i for i+1 = c0*1+c1*2+c2*4
        //correct X_i for i+1 = c3*1+c4*2+c5*4
        for (int j = 0; j < 7; j++) {
            writeClassicalControl(dd::Qubit(clAncStart), dd::QubitCount(3), j + 1U, xSyndrome ? qc::Z : qc::X, i + j * nQubits);
        }
    }
    gatesWritten = true;
}

void Q7SteaneEcc::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const int    nQubits             = qc.getNqubits();
    const int    clAncStart          = static_cast<int>(qc.getNcbits());
    unsigned int correction_needed[] = {1, 2, 4, 7}; //values with odd amount of '1' bits
    //use exiting registers qeccX and qeccZ for decoding

    for (int i = 0; i < nQubits; i++) {
        //#|###|###
        //0|111|111
        //odd amount of 1's -> x[0] = 1
        //measure from index 1 (not 0) to 6, =qubit 2 to 7

        qcMapped.measure(static_cast<dd::Qubit>(i + 1 * nQubits), clAncStart);
        qcMapped.measure(static_cast<dd::Qubit>(i + 2 * nQubits), clAncStart + 1);
        qcMapped.measure(static_cast<dd::Qubit>(i + 3 * nQubits), clAncStart + 2);
        for (unsigned int value: correction_needed) {
            writeClassicalControl(dd::Qubit(clAncStart), dd::QubitCount(3), value, qc::X, i);
        }
        qcMapped.measure(static_cast<dd::Qubit>(i + 4 * nQubits), clAncStart);
        qcMapped.measure(static_cast<dd::Qubit>(i + 5 * nQubits), clAncStart + 1);
        qcMapped.measure(static_cast<dd::Qubit>(i + 6 * nQubits), clAncStart + 2);
        for (unsigned int value: correction_needed) {
            writeClassicalControl(dd::Qubit(clAncStart), dd::QubitCount(3), value, qc::X, i);
        }
    }
    isDecoded = true;
}

void Q7SteaneEcc::mapGate(const std::unique_ptr<qc::Operation>& gate, qc::QuantumComputation& qc) {
    if (isDecoded && gate->getType() != qc::Measure) {
        writeEncoding();
    }
    const int                nQubits = qc.getNqubits();
    qc::NonUnitaryOperation* measureGate;
    switch (gate->getType()) {
        case qc::I:
            break;
        case qc::X:
        case qc::H:
        case qc::Y:
        case qc::Z:
            for (std::size_t t = 0; t < gate->getNtargets(); t++) {
                int i = static_cast<unsigned char>(gate->getTargets()[t]);
                if (gate->getNcontrols() == 2 && decomposeMultiControlledGates) {
                    auto& ctrls     = gate->getControls();
                    int   idx       = 0;
                    int   ctrl2[2]  = {-1, -1};
                    bool  ctrl2T[2] = {true, true};
                    for (const auto& ct: ctrls) {
                        ctrl2[idx]  = static_cast<unsigned char>(ct.qubit);
                        ctrl2T[idx] = ct.type == dd::Control::Type::pos;
                        idx++;
                    }
                    if (gate->getType() == qc::X) {
                        for (int j = 0; j < 7; j++) {
                            writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits,
                                         ctrl2T[1]);
                        }
                    } else if (gate->getType() == qc::Z) {
                        for (int j = 0; j < 7; j++) {
                            qcMapped.h(static_cast<dd::Qubit>(i + j * nQubits));
                            writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits,
                                         ctrl2T[1]);
                            qcMapped.h(static_cast<dd::Qubit>(i + j * nQubits));
                        }
                    } else if (gate->getType() == qc::Y) {
                        for (int j = 0; j < 7; j++) {
                            writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits,
                                         ctrl2T[1]);
                            qcMapped.h(static_cast<dd::Qubit>(i + j * nQubits));
                            writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits,
                                         ctrl2T[1]);
                            qcMapped.h(static_cast<dd::Qubit>(i + j * nQubits));
                        }
                    } else {
                        gateNotAvailableError(gate);
                    }
                } else if (gate->getNcontrols() > 2 && decomposeMultiControlledGates) {
                    gateNotAvailableError(gate);
                } else if (gate->getNcontrols()) {
                    auto& ctrls = gate->getControls();
                    for (int j = 0; j < 7; j++) {
                        dd::Controls ctrls2;
                        for (const auto& ct: ctrls) {
                            ctrls2.insert(dd::Control{dd::Qubit(ct.qubit + j * nQubits), ct.type});
                        }
                        //qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls2, i+j*nQubits, gate->getType());
                        writeGeneric(static_cast<dd::Qubit>(i + j * nQubits), ctrls2, gate->getType());
                    }
                } else {
                    for (int j = 0; j < 7; j++) {
                        writeGeneric(static_cast<dd::Qubit>(i + j * nQubits), gate->getType());
                        //qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i+j*nQubits, gate->getType());
                    }
                }
            }
            break;
            //locigal S = 3 physical S's
        case qc::S:
        case qc::Sdag:
            for (std::size_t t = 0; t < gate->getNtargets(); t++) {
                int i = static_cast<unsigned char>(gate->getTargets()[t]);
                if (gate->getNcontrols() > 1 && decomposeMultiControlledGates) {
                    gateNotAvailableError(gate);
                } else if (gate->getNcontrols() && cliffordGatesOnly) {
                    gateNotAvailableError(gate);
                } else if (gate->getNcontrols()) {
                    auto& ctrls = gate->getControls();
                    for (int j = 0; j < 7; j++) {
                        dd::Controls ctrls2;
                        for (const auto& ct: ctrls) {
                            ctrls2.insert(dd::Control{dd::Qubit(ct.qubit + j * nQubits), ct.type});
                        }
                        writeGeneric(i + j * nQubits, ctrls2, gate->getType());
                        writeGeneric(i + j * nQubits, ctrls2, gate->getType());
                        writeGeneric(i + j * nQubits, ctrls2, gate->getType());
                    }
                } else {
                    if (gate->getType() == qc::S) {
                        for (int j = 0; j < 7; j++) {
                            qcMapped.s(static_cast<dd::Qubit>(i + j * nQubits));
                            qcMapped.s(static_cast<dd::Qubit>(i + j * nQubits));
                            qcMapped.s(static_cast<dd::Qubit>(i + j * nQubits));
                        }
                    } else {
                        for (int j = 0; j < 7; j++) {
                            writeSdag(static_cast<dd::Qubit>(i + j * nQubits));
                            writeSdag(static_cast<dd::Qubit>(i + j * nQubits));
                            writeSdag(static_cast<dd::Qubit>(i + j * nQubits));
                        }
                    }
                }
            }
            break;
        case qc::Measure:
            if (!isDecoded) {
                measureAndCorrect();
                writeDecoding();
            }
            measureGate = (qc::NonUnitaryOperation*)gate.get();
            for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
                auto classicalRegisterName = qc.returnClassicalRegisterName(measureGate->getTargets()[j]);
                if (!classicalRegisterName.empty()) {
                    qcMapped.measure(static_cast<dd::Qubit>(measureGate->getClassics()[j]), {classicalRegisterName, measureGate->getTargets()[j]});
                } else {
                    qcMapped.measure(static_cast<dd::Qubit>(measureGate->getClassics()[j]), measureGate->getTargets()[j]);
                }
            }
            break;
        case qc::T:
        case qc::Tdag:
            for (std::size_t t = 0; t < gate->getNtargets(); t++) {
                int i = static_cast<unsigned char>(gate->getTargets()[t]);
                if (gate->getControls().empty()) {
                    //arXiv:1610.03309v1
                    writeX(dd::Qubit(i + 5 * nQubits), dd::Control{dd::Qubit(i + 6 * nQubits), dd::Control::Type::pos});
                    writeX(dd::Qubit(i + 0 * nQubits), dd::Control{dd::Qubit(i + 5 * nQubits), dd::Control::Type::pos});
                    if (gate->getType() == qc::T) {
                        qcMapped.t(dd::Qubit(i + 0 * nQubits));
                    } else {
                        qcMapped.tdag(dd::Qubit(i + 0 * nQubits));
                    }
                    writeX(dd::Qubit(i + 0 * nQubits), dd::Control{dd::Qubit(i + 5 * nQubits), dd::Control::Type::pos});
                    writeX(dd::Qubit(i + 5 * nQubits), dd::Control{dd::Qubit(i + 6 * nQubits), dd::Control::Type::pos});
                } else {
                    gateNotAvailableError(gate);
                }
            }
            break;
        default:
            gateNotAvailableError(gate);
    }
}
