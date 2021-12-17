/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q3ShorEcc.hpp"

//3 data qubits, 2 for measuring -> 5 qubits per physical qubit
Q3ShorEcc::Q3ShorEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly): Ecc({ID::Q3Shor, 3, 2, Q3ShorEcc::getName()}, qc, measureFq, decomposeMC, cliffOnly) {}

void Q3ShorEcc::initMappedCircuit() {
//method is overridden because we need 2 kinds of classical measurement output registers
    qc.stripIdleQubits(true, false);
    statistics.nInputQubits = qc.getNqubits();
    statistics.nInputClassicalBits = (int)qc.getNcbits();
	statistics.nOutputQubits = qc.getNqubits()*ecc.nRedundantQubits+ecc.nCorrectingBits;
	statistics.nOutputClassicalBits = statistics.nInputClassicalBits+ecc.nCorrectingBits;
	qcMapped.addQubitRegister(statistics.nOutputQubits);
	qcMapped.addClassicalRegister(statistics.nInputClassicalBits);
	qcMapped.addClassicalRegister(2, "qecc");
}

void Q3ShorEcc::writeEncoding() {
	const int nQubits = qc.getNqubits();

    for(int i=0;i<nQubits;i++) {
        auto ctrl = dd::Control{dd::Qubit(i), dd::Control::Type::pos};
        qcMapped.x(i+nQubits, ctrl);
        qcMapped.x(i+2*nQubits, ctrl);
    }
    decodingDone = false;
}

void Q3ShorEcc::measureAndCorrect() {
    const int nQubits = qc.getNqubits();
    const int ancStart = nQubits * ecc.nRedundantQubits; //measure start (index of first ancilla qubit)
    const int clStart = statistics.nInputClassicalBits;
    for(int i=0;i<nQubits;i++) {
        qcMapped.reset(ancStart);
        qcMapped.reset(ancStart+1);
        qcMapped.h(ancStart);
        qcMapped.h(ancStart+1);
        auto a0 = dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos};
        auto a1 = dd::Control{dd::Qubit(ancStart+1), dd::Control::Type::pos};
        qcMapped.z(i,           a0);
        qcMapped.z(i+nQubits,   a0);
        qcMapped.z(i+nQubits,   a1);
        qcMapped.z(i+2*nQubits, a1);
        qcMapped.h(ancStart);
        qcMapped.h(ancStart+1);

        qcMapped.measure(ancStart, clStart);
        qcMapped.measure(ancStart+1, clStart+1);

        std::unique_ptr<qc::Operation> op1 = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(), dd::Qubit(i), qc::X);
        const auto pair1 = std::make_pair(dd::Qubit(clStart), dd::QubitCount(2));
        qcMapped.emplace_back<qc::ClassicControlledOperation>(op1, pair1, 1);

        std::unique_ptr<qc::Operation> op2 = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(), dd::Qubit(i+2*nQubits), qc::X);
        const auto pair2 = std::make_pair(dd::Qubit(clStart), dd::QubitCount(2));
        qcMapped.emplace_back<qc::ClassicControlledOperation>(op2, pair2, 2);

        std::unique_ptr<qc::Operation> op3 = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(), dd::Qubit(i+nQubits), qc::X);
        const auto pair3 = std::make_pair(dd::Qubit(clStart), dd::QubitCount(2));
        qcMapped.emplace_back<qc::ClassicControlledOperation>(op3, pair3, 3);
    }
}

void Q3ShorEcc::writeDecoding() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        auto ctrl = dd::Control{dd::Qubit(i), dd::Control::Type::pos};
        qcMapped.x(i+nQubits, ctrl);
        qcMapped.x(i+2*nQubits, ctrl);

        writeToffoli(i, i+nQubits, true, i+2*nQubits, true);
    }
    decodingDone = true;
}

void Q3ShorEcc::mapGate(const std::unique_ptr<qc::Operation> &gate) {
    if(decodingDone && gate.get()->getType()!=qc::Measure && gate.get()->getType()!=qc::H) {
        writeEncoding();
    }
    const int nQubits = qc.getNqubits();
    qc::NonUnitaryOperation *measureGate=nullptr;
    switch(gate.get()->getType()) {
    case qc::I: break;
    case qc::X:
    //case qc::H:
    case qc::Y:
    case qc::Z:
    case qc::S:
    case qc::Sdag:
    case qc::T:
    case qc::Tdag:
        for(std::size_t j=0;j<gate.get()->getNtargets();j++) {
            auto i = gate.get()->getTargets()[j];
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
                    writeToffoli(i, ctrl2[0], ctrl2T[0], ctrl2[1], ctrl2T[1]);
                    writeToffoli(i+nQubits, ctrl2[0]+nQubits, ctrl2T[0], ctrl2[1]+nQubits, ctrl2T[1]);
                    writeToffoli(i+2*nQubits, ctrl2[0]+2*nQubits, ctrl2T[0], ctrl2[1]+2*nQubits, ctrl2T[1]);
                } else if(gate.get()->getType()==qc::Z) {
                    qcMapped.h(i);
                    qcMapped.h(i+nQubits);
                    qcMapped.h(i+2*nQubits);
                    writeToffoli(i, ctrl2[0], ctrl2T[0], ctrl2[1], ctrl2T[1]);
                    writeToffoli(i+nQubits, ctrl2[0]+nQubits, ctrl2T[0], ctrl2[1]+nQubits, ctrl2T[1]);
                    writeToffoli(i+2*nQubits, ctrl2[0]+2*nQubits, ctrl2T[0], ctrl2[1]+2*nQubits, ctrl2T[1]);
                    qcMapped.h(i);
                    qcMapped.h(i+nQubits);
                    qcMapped.h(i+2*nQubits);
                } else if(gate.get()->getType()==qc::Y) {
                    writeToffoli(i, ctrl2[0], ctrl2T[0], ctrl2[1], ctrl2T[1]);
                    writeToffoli(i+nQubits, ctrl2[0]+nQubits, ctrl2T[0], ctrl2[1]+nQubits, ctrl2T[1]);
                    writeToffoli(i+2*nQubits, ctrl2[0]+2*nQubits, ctrl2T[0], ctrl2[1]+2*nQubits, ctrl2T[1]);
                    qcMapped.h(i);
                    qcMapped.h(i+nQubits);
                    qcMapped.h(i+2*nQubits);
                    writeToffoli(i, ctrl2[0], ctrl2T[0], ctrl2[1], ctrl2T[1]);
                    writeToffoli(i+nQubits, ctrl2[0]+nQubits, ctrl2T[0], ctrl2[1]+nQubits, ctrl2T[1]);
                    writeToffoli(i+2*nQubits, ctrl2[0]+2*nQubits, ctrl2T[0], ctrl2[1]+2*nQubits, ctrl2T[1]);
                    qcMapped.h(i);
                    qcMapped.h(i+nQubits);
                    qcMapped.h(i+2*nQubits);
                } else {
                    gateNotAvailableError(gate);
                }
            } else if(gate.get()->getNcontrols()>2 && decomposeMultiControlledGates) {
                gateNotAvailableError(gate);
            } else if(gate.get()->getNcontrols()) {
                auto& ctrls = gate.get()->getControls();
                qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls, i, gate.get()->getType());
                dd::Controls ctrls2, ctrls3;
                for(const auto &ct: ctrls) {
                    ctrls2.insert(dd::Control{dd::Qubit(ct.qubit+nQubits), ct.type});
                    ctrls3.insert(dd::Control{dd::Qubit(ct.qubit+2*nQubits), ct.type});
                }
                qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls2, i+nQubits, gate.get()->getType());
                qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls3, i+2*nQubits, gate.get()->getType());
            } else {
                qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i, gate.get()->getType());
                qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i+nQubits, gate.get()->getType());
                qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i+2*nQubits, gate.get()->getType());
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
            qcMapped.measure(measureGate->getTargets()[j], measureGate->getClassics()[j]);
        }
        break;
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
        break;
    }
}
