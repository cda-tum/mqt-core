/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q9ShorEcc.hpp"

//9 data qubits, 8 for measuring -> 17 qubits per physical qubit
Q9ShorEcc::Q9ShorEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly):
    Ecc({ID::Q9Shor, 9, 8, Q9ShorEcc::getName()}, qc, measureFq, decomposeMC, cliffOnly) {}

void Q9ShorEcc::initMappedCircuit() {
    //method is overridden because we need 2 kinds of classical measurement output registers
    qc.stripIdleQubits(true, false);
    statistics.nInputQubits         = qc.getNqubits();
    statistics.nInputClassicalBits  = (int)qc.getNcbits();
    statistics.nOutputQubits        = qc.getNqubits() * ecc.nRedundantQubits + ecc.nCorrectingBits;
    statistics.nOutputClassicalBits = statistics.nInputClassicalBits + ecc.nCorrectingBits;
    qcMapped.addQubitRegister(statistics.nOutputQubits);
    qcMapped.addClassicalRegister(statistics.nInputClassicalBits);
    qcMapped.addClassicalRegister(2, "qeccX1");
    qcMapped.addClassicalRegister(2, "qeccX2");
    qcMapped.addClassicalRegister(2, "qeccX3");
    qcMapped.addClassicalRegister(2, "qeccZ");
}

void Q9ShorEcc::writeEncoding() {
    if (!isDecoded) {
        return;
    }
    isDecoded         = false;
    const int nQubits = qc.getNqubits();
    for (int i = 0; i < nQubits; i++) {
        dd::Control ci = {dd::Qubit(i), dd::Control::Type::pos};
        writeX(dd::Qubit(i + 3 * nQubits), ci);
        writeX(dd::Qubit(i + 6 * nQubits), ci);

        qcMapped.h(dd::Qubit(i));
        qcMapped.h(dd::Qubit(i + 3 * nQubits));
        qcMapped.h(dd::Qubit(i + 6 * nQubits));

        dd::Control ci3 = {dd::Qubit(i + 3 * nQubits), dd::Control::Type::pos};
        dd::Control ci6 = {dd::Qubit(i + 6 * nQubits), dd::Control::Type::pos};
        writeX(dd::Qubit(i + nQubits), ci);
        writeX(dd::Qubit(i + 2 * nQubits), ci);
        writeX(dd::Qubit(i + 4 * nQubits), ci3);
        writeX(dd::Qubit(i + 5 * nQubits), ci3);
        writeX(dd::Qubit(i + 7 * nQubits), ci6);
        writeX(dd::Qubit(i + 8 * nQubits), ci6);
    }
}

void Q9ShorEcc::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    const int nQubits = qc.getNqubits();
    const int clStart = statistics.nInputClassicalBits;
    for (int i = 0; i < nQubits; i++) {
        //syntactic sugar for qubit indices
        dd::Qubit   q[9];   //qubits
        dd::Qubit   a[8];   //ancilla qubits
        dd::Control ca[8];  //ancilla controls
        dd::Control cna[8]; //negative ancilla controls
        for (int j = 0; j < 9; j++) { q[j] = dd::Qubit(i + j * nQubits); }
        for (int j = 0; j < 8; j++) {
            a[j] = dd::Qubit(ecc.nRedundantQubits * nQubits + j);
            qcMapped.reset(a[j]);
        }
        for (int j = 0; j < 8; j++) { ca[j] = dd::Control{dd::Qubit(a[j]), dd::Control::Type::pos}; }
        for (int j = 0; j < 8; j++) { cna[j] = dd::Control{dd::Qubit(a[j]), dd::Control::Type::neg}; }

        // PREPARE measurements --------------------------------------------------------
        for (dd::Qubit j: a) {
            qcMapped.h(j);
        }
        //x errors = indirectly via controlled z
        writeZ(q[0], ca[0]);
        writeZ(q[1], ca[0]);
        writeZ(q[1], ca[1]);
        writeZ(q[2], ca[1]);

        writeZ(q[3], ca[2]);
        writeZ(q[4], ca[2]);
        writeZ(q[4], ca[3]);
        writeZ(q[5], ca[3]);

        writeZ(q[6], ca[4]);
        writeZ(q[7], ca[4]);
        writeZ(q[7], ca[5]);
        writeZ(q[8], ca[5]);

        //z errors = indirectly via controlled x/CNOT
        writeX(q[0], ca[6]);
        writeX(q[1], ca[6]);
        writeX(q[2], ca[6]);
        writeX(q[3], ca[6]);
        writeX(q[4], ca[6]);
        writeX(q[5], ca[6]);

        writeX(q[3], ca[7]);
        writeX(q[4], ca[7]);
        writeX(q[5], ca[7]);
        writeX(q[6], ca[7]);
        writeX(q[7], ca[7]);
        writeX(q[8], ca[7]);

        for (dd::Qubit j: a) { qcMapped.h(j); }

        //MEASURE ancilla qubits
        for (int j = 0; j < 8; j++) {
            qcMapped.measure(a[j], clStart + j);
        }

        //CORRECT
        //x, i.e. bit flip errors
        writeClassicalControl(dd::Qubit(clStart), 2, 1, qc::X, i);
        writeClassicalControl(dd::Qubit(clStart), 2, 2, qc::X, i + 2 * nQubits);
        writeClassicalControl(dd::Qubit(clStart), 2, 3, qc::X, i + nQubits);

        writeClassicalControl(dd::Qubit(clStart + 2), 2, 1, qc::X, i + 3 * nQubits);
        writeClassicalControl(dd::Qubit(clStart + 2), 2, 2, qc::X, i + 5 * nQubits);
        writeClassicalControl(dd::Qubit(clStart + 2), 2, 3, qc::X, i + 4 * nQubits);

        writeClassicalControl(dd::Qubit(clStart + 4), 2, 1, qc::X, i + 6 * nQubits);
        writeClassicalControl(dd::Qubit(clStart + 4), 2, 2, qc::X, i + 8 * nQubits);
        writeClassicalControl(dd::Qubit(clStart + 4), 2, 3, qc::X, i + 7 * nQubits);

        //z, i.e. phase flip errors
        writeClassicalControl(dd::Qubit(clStart + 6), 2, 1, qc::Z, i);
        writeClassicalControl(dd::Qubit(clStart + 6), 2, 2, qc::Z, i + 6 * nQubits);
        writeClassicalControl(dd::Qubit(clStart + 6), 2, 3, qc::Z, i + 3 * nQubits);
    }
}

void Q9ShorEcc::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const int nQubits = qc.getNqubits();
    for (int i = 0; i < nQubits; i++) {
        dd::Control ci[9];
        for (int j = 0; j < 9; j++) {
            ci[j] = dd::Control{dd::Qubit(i + j * nQubits), dd::Control::Type::pos};
        }

        writeX(i + nQubits, ci[0]);
        writeX(i + 2 * nQubits, ci[0]);

        writeX(i + 4 * nQubits, ci[3]);
        writeX(i + 5 * nQubits, ci[3]);

        writeX(i + 7 * nQubits, ci[6]);
        writeX(i + 8 * nQubits, ci[6]);

        writeToffoli(i, i + nQubits, true, i + 2 * nQubits, true);
        writeToffoli(i + 3 * nQubits, i + 4 * nQubits, true, i + 5 * nQubits, true);
        writeToffoli(i + 6 * nQubits, i + 7 * nQubits, true, i + 8 * nQubits, true);

        qcMapped.h(i);
        qcMapped.h(i + 3 * nQubits);
        qcMapped.h(i + 6 * nQubits);

        writeX(i + 3 * nQubits, ci[0]);
        writeX(i + 6 * nQubits, ci[0]);
        writeToffoli(i, i + 3 * nQubits, true, i + 6 * nQubits, true);
    }
    isDecoded = true;
}

void Q9ShorEcc::mapGate(const std::unique_ptr<qc::Operation>& gate) {
    if (isDecoded && gate.get()->getType() != qc::Measure && gate.get()->getType() != qc::H) {
        writeEncoding();
    }
    const int                nQubits     = qc.getNqubits();
    qc::NonUnitaryOperation* measureGate = nullptr;
    int                      i;
    auto                     type = qc::I;
    switch (gate.get()->getType()) {
        case qc::I: break;
        case qc::X:
            type = qc::Z;
            break;
        case qc::Y:
            type = qc::Y;
            break;
        case qc::Z:
            type = qc::X;
            break;
        case qc::Measure:
            if (!isDecoded) {
                measureAndCorrect();
                writeDecoding();
            }
            measureGate = (qc::NonUnitaryOperation*)gate.get();
            for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
                qcMapped.measure(measureGate->getTargets()[j], measureGate->getClassics()[j]);
            }
            return;
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
    for (std::size_t t = 0; t < gate.get()->getNtargets(); t++) {
        i = gate.get()->getTargets()[t];
        if (gate.get()->getNcontrols() == 2 && decomposeMultiControlledGates) {
            //Q9Shor code: put H gate before and after each control point, i.e. "cx 0,1" becomes "h0; cz 0,1; h0"
            auto& ctrls     = gate.get()->getControls();
            int   idx       = 0;
            int   ctrl2[2]  = {-1, -1};
            bool  ctrl2T[2] = {true, true};
            for (const auto& ct: ctrls) {
                ctrl2[idx]  = ct.qubit;
                ctrl2T[idx] = ct.type == dd::Control::Type::pos;
                idx++;
            }
            for (int j = 0; j < 9; j++) {
                qcMapped.h(ctrl2[0] + j * nQubits);
                qcMapped.h(ctrl2[1] + j * nQubits);
            }
            if (type == qc::X) {
                for (int j = 0; j < 9; j++) {
                    writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits, ctrl2T[1]);
                }
            } else if (type == qc::Z) {
                for (int j = 0; j < 9; j++) {
                    qcMapped.h(i + j * nQubits);
                    writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits, ctrl2T[1]);
                    qcMapped.h(i + j * nQubits);
                }
            } else if (type == qc::Y) {
                for (int j = 0; j < 9; j++) {
                    writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits, ctrl2T[1]);
                    qcMapped.h(i + j * nQubits);
                    writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits, ctrl2T[1]);
                    qcMapped.h(i + j * nQubits);
                }
            } else {
                gateNotAvailableError(gate);
            }
            for (int j = 0; j < 9; j++) {
                qcMapped.h(ctrl2[0] + j * nQubits);
                qcMapped.h(ctrl2[1] + j * nQubits);
            }

        } else if (gate.get()->getNcontrols() > 2 && decomposeMultiControlledGates) {
            gateNotAvailableError(gate);
        } else if (gate.get()->getNcontrols()) {
            //Q9Shor code: put H gate before and after each control point, i.e. "cx 0,1" becomes "h0; cz 0,1; h0"
            auto& ctrls = gate.get()->getControls();
            for (int j = 0; j < 9; j++) {
                dd::Controls ctrls2;
                for (const auto& ct: ctrls) {
                    ctrls2.insert(dd::Control{dd::Qubit(ct.qubit + j * nQubits), ct.type});
                    qcMapped.h(ct.qubit + j * nQubits);
                }
                writeGeneric(i + j * nQubits, ctrls2, type);
                for (const auto& ct: ctrls) {
                    qcMapped.h(ct.qubit + j * nQubits);
                }
            }
        } else {
            for (int j = 0; j < 9; j++) {
                writeGeneric(i + j * nQubits, type);
            }
        }
    }
}
