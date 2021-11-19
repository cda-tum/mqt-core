/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q5LaflammeEcc.hpp"

//5 data qubits, 4 for measuring
Q5LaflammeEcc::Q5LaflammeEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC): Ecc({ID::Q5Laflamme, 5, 4, Q5LaflammeEcc::getName()}, qc, measureFq, decomposeMC) {}

void Q5LaflammeEcc::initMappedCircuit() {
//method is overridden because we need 2 kinds of classical measurement output registers
    qc.stripIdleQubits(true, false);
    statistics.nInputQubits = qc.getNqubits();
    statistics.nInputClassicalBits = (int)qc.getNcbits();
	statistics.nOutputQubits = qc.getNqubits()*ecc.nRedundantQubits+ecc.nCorrectingBits;
	statistics.nOutputClassicalBits = statistics.nInputClassicalBits+ecc.nCorrectingBits;
	qcMapped.addQubitRegister(statistics.nOutputQubits);
	qcMapped.addClassicalRegister(statistics.nInputClassicalBits);
	qcMapped.addClassicalRegister(4, "qecc");
}

void Q5LaflammeEcc::writeEncoding() {
    measureAndCorrect();
    decodingDone = false;
}

void Q5LaflammeEcc::measureAndCorrect() {
    const int nQubits = qc.getNqubits();
    const int ancStart = nQubits * ecc.nRedundantQubits;
    const int clAncStart = qc.getNcbits();


    for(int i=0;i<nQubits;i++) {
        int q[5] = {};
        for(int j=0;j<5;j++) {q[j] = i+j*nQubits;}

        qcMapped.reset(ancStart);
        qcMapped.reset(ancStart+1);
        qcMapped.reset(ancStart+2);
        qcMapped.reset(ancStart+3);

        qcMapped.h(ancStart);
        qcMapped.h(ancStart+1);
        qcMapped.h(ancStart+2);
        qcMapped.h(ancStart+3);

        auto c0 = dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos};
        auto c1 = dd::Control{dd::Qubit(ancStart+1), dd::Control::Type::pos};
        auto c2 = dd::Control{dd::Qubit(ancStart+2), dd::Control::Type::pos};
        auto c3 = dd::Control{dd::Qubit(ancStart+3), dd::Control::Type::pos};

        //traversal of matrix: "/"
        //K1: XZZXI
        //K2: IXZZX
        //K3: XIXZZ
        //K4: ZXIXZ

        qcMapped.x(q[0], c0);

        qcMapped.z(q[1], c0);
        //controlled-id(i, c1)

        qcMapped.z(q[2], c0);
        qcMapped.x(q[1], c1);
        qcMapped.x(q[0], c2);

        qcMapped.x(q[3], c0);
        qcMapped.z(q[2], c1);
        //controlled-id(i+1, c2)
        qcMapped.z(q[0], c3);

        //controlled-id(i+4, c0)
        qcMapped.z(q[3], c1);
        qcMapped.x(q[2], c2);
        qcMapped.x(q[1], c3);

        qcMapped.x(q[4], c1);
        qcMapped.z(q[3], c2);
        //controlled-id(i+2, c3)

        qcMapped.z(q[4], c2);
        qcMapped.x(q[3], c3);

        qcMapped.z(q[4], c3);

        qcMapped.h(ancStart);
        qcMapped.h(ancStart+1);
        qcMapped.h(ancStart+2);
        qcMapped.h(ancStart+3);

        qcMapped.measure(ancStart, clAncStart);
        qcMapped.measure(ancStart+1, clAncStart+1);
        qcMapped.measure(ancStart+2, clAncStart+2);
        qcMapped.measure(ancStart+3, clAncStart+3);


        writeClassicalControlled(1, q[1], qc::X);
        writeClassicalControlled(2, q[4], qc::Z);
        writeClassicalControlled(3, q[2], qc::X);
        writeClassicalControlled(4, q[2], qc::Z);
        writeClassicalControlled(5, q[0], qc::Z);
        writeClassicalControlled(6, q[3], qc::X);
        writeClassicalControlled(7, q[2], qc::Y);
        writeClassicalControlled(8, q[0], qc::X);
        writeClassicalControlled(9, q[3], qc::Z);
        writeClassicalControlled(10, q[1], qc::Z);
        writeClassicalControlled(11, q[1], qc::Y);
        writeClassicalControlled(12, q[4], qc::X);
        writeClassicalControlled(13, q[0], qc::Y);
        writeClassicalControlled(14, q[4], qc::Y);
        writeClassicalControlled(15, q[3], qc::Y);
    }
}

void Q5LaflammeEcc::writeClassicalControlled(const unsigned int value, int target, qc::OpType optype) {
    std::unique_ptr<qc::Operation> op = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(), target, optype);
    const auto pair_ = std::make_pair(dd::Qubit(statistics.nInputClassicalBits), dd::QubitCount(4));
    qcMapped.emplace_back<qc::ClassicControlledOperation>(op, pair_, value);
}

void Q5LaflammeEcc::writeDecoding() {
    const int nQubits = qc.getNqubits();
    const int ancStart = nQubits * ecc.nRedundantQubits;

    for(int i=0;i<nQubits;i++) {
        qcMapped.reset(ancStart);
        qcMapped.h(ancStart);

        auto c = dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos};
        for(int j=0;j<5;j++) {
            qcMapped.z(i+j*nQubits, c);
        }
        qcMapped.h(ancStart);

        qcMapped.measure(ancStart, i);
        qcMapped.swap(ancStart, i);
    }
    decodingDone = true;
}

void Q5LaflammeEcc::mapGate(const std::unique_ptr<qc::Operation> &gate) {
    if(decodingDone && gate.get()->getType()!=qc::Measure) {
        writeEncoding();
    }
    const int nQubits = qc.getNqubits();
    qc::NonUnitaryOperation *measureGate=nullptr;
    switch(gate.get()->getType()) {
    case qc::I: break;
    case qc::X:
    case qc::H:
    case qc::Y:
    case qc::Z:
        for(std::size_t t=0;t<gate.get()->getNtargets();t++) {
            int i = gate.get()->getTargets()[t];
            if(gate.get()->getNcontrols()==2 && decomposeMultiControlledGates) {

                auto& ctrls = gate.get()->getControls();
                int idx=0;
                int ctrl2[2] = {-1, -1};
                bool ctrl2T[2] = {true, true};
                for(const auto &ct: ctrls) {
                    ctrl2[idx] = ct.qubit;
                    ctrl2T[idx] = ct.type == dd::Control::Type::pos;
                    idx++;
                }
                if(gate.get()->getType()==qc::X) {
                    for(int j=0;j<5;j++) {
                        writeToffoli(i+j*nQubits, ctrl2[0]+j*nQubits, ctrl2T[0], ctrl2[1]+j*nQubits, ctrl2T[1]);
                    }
                } else if(gate.get()->getType()==qc::Z) {
                    for(int j=0;j<5;j++) {
                        qcMapped.h(i+j*nQubits);
                        writeToffoli(i+j*nQubits, ctrl2[0]+j*nQubits, ctrl2T[0], ctrl2[1]+j*nQubits, ctrl2T[1]);
                        qcMapped.h(i+j*nQubits);
                    }
                } else if(gate.get()->getType()==qc::Y) {
                    for(int j=0;j<5;j++) {
                        writeToffoli(i+j*nQubits, ctrl2[0]+j*nQubits, ctrl2T[0], ctrl2[1]+j*nQubits, ctrl2T[1]);
                        qcMapped.h(i+j*nQubits);
                        writeToffoli(i+j*nQubits, ctrl2[0]+j*nQubits, ctrl2T[0], ctrl2[1]+j*nQubits, ctrl2T[1]);
                        qcMapped.h(i+j*nQubits);
                    }
                } else {
                    gateNotAvailableError(gate);
                }
            } else if(gate.get()->getNcontrols()>2 && decomposeMultiControlledGates) {
                gateNotAvailableError(gate);
            } else if(gate.get()->getNcontrols()) {
                auto& ctrls = gate.get()->getControls();
                for(int j=0;j<5;j++) {
                    dd::Controls ctrls2;
                    for(const auto &ct: ctrls) {
                        ctrls2.insert(dd::Control{dd::Qubit(ct.qubit+j*nQubits), ct.type});
                    }
                    qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls2, i+j*nQubits, gate.get()->getType());
                }
            } else {
                for(int j=0;j<5;j++) {
                    qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i+j*nQubits, gate.get()->getType());
                }
            }
        }
        break;
    case qc::Measure:
        if(!decodingDone) {
            measureAndCorrect();
            writeDecoding();
        }
        measureGate = (qc::NonUnitaryOperation*)gate.get();
        for(std::size_t j=0;j<measureGate->getNclassics();j++) {
            qcMapped.measure(measureGate->getClassics()[j], measureGate->getTargets()[j]);
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

