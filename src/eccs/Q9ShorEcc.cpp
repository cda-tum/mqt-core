/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q9ShorEcc.hpp"

//9 data qubits, 8 for measuring -> 17 qubits per physical qubit
Q9ShorEcc::Q9ShorEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly) : Ecc({ID::Q9Shor, 9, 8, Q9ShorEcc::getName()}, qc, measureFq, decomposeMC, cliffOnly) {}

void Q9ShorEcc::initMappedCircuit() {
//method is overridden because we need 2 kinds of classical measurement output registers
    qc.stripIdleQubits(true, false);
    statistics.nInputQubits = qc.getNqubits();
    statistics.nInputClassicalBits = (int)qc.getNcbits();
	statistics.nOutputQubits = qc.getNqubits()*ecc.nRedundantQubits+ecc.nCorrectingBits;
	statistics.nOutputClassicalBits = statistics.nInputClassicalBits+ecc.nCorrectingBits;
	qcMapped.addQubitRegister(statistics.nOutputQubits);
	qcMapped.addClassicalRegister(statistics.nInputClassicalBits);
	qcMapped.addClassicalRegister(2, "qeccX1");
	qcMapped.addClassicalRegister(2, "qeccX2");
	qcMapped.addClassicalRegister(2, "qeccX3");
	qcMapped.addClassicalRegister(2, "qeccZ");
}

void Q9ShorEcc::writeEncoding() {
	const int nQubits = qc.getNqubits();
	for(int i=0;i<nQubits;i++) {
        dd::Control ci = {dd::Qubit(i), dd::Control::Type::pos};
        qcMapped.x(i+3*nQubits, ci);
        qcMapped.x(i+6*nQubits, ci);

        qcMapped.h(i);
        qcMapped.h(i+3*nQubits);
        qcMapped.h(i+6*nQubits);

        dd::Control ci3 = {dd::Qubit(i+3*nQubits), dd::Control::Type::pos};
        dd::Control ci6 = {dd::Qubit(i+6*nQubits), dd::Control::Type::pos};
        qcMapped.x(i+nQubits, ci);
        qcMapped.x(i+2*nQubits, ci);
        qcMapped.x(i+4*nQubits, ci3);
        qcMapped.x(i+5*nQubits, ci3);
        qcMapped.x(i+7*nQubits, ci6);
        qcMapped.x(i+8*nQubits, ci6);
    }
    decodingDone = false;
}

void Q9ShorEcc::measureAndCorrect() {
    const int nQubits = qc.getNqubits();
    const int clStart = statistics.nInputClassicalBits;
    for(int i=0;i<nQubits;i++) {
        //syntactic sugar for qubit indices
        unsigned int q[9];//qubits
        unsigned int a[8];//ancilla qubits
        dd::Control ca[8];//ancilla controls
        dd::Control cna[8];//negative ancilla controls
        for(int j=0;j<9;j++) { q[j] = i+j*nQubits;}
        for(int j=0;j<8;j++) { a[j] = ecc.nRedundantQubits*nQubits+j; qcMapped.reset(a[j]); }
        for(int j=0;j<8;j++) { ca[j] = dd::Control{dd::Qubit(a[j]), dd::Control::Type::pos}; }
        for(int j=0;j<8;j++) { cna[j] = dd::Control{dd::Qubit(a[j]), dd::Control::Type::neg}; }


        // PREPARE measurements --------------------------------------------------------
        for(int j=0;j<8;j++) {
            qcMapped.h(a[j]);
        }
        //x errors = indirectly via controlled z
        qcMapped.z(q[0], ca[0]);
        qcMapped.z(q[1], ca[0]);
        qcMapped.z(q[1], ca[1]);
        qcMapped.z(q[2], ca[1]);

        qcMapped.z(q[3], ca[2]);
        qcMapped.z(q[4], ca[2]);
        qcMapped.z(q[4], ca[3]);
        qcMapped.z(q[5], ca[3]);

        qcMapped.z(q[6], ca[4]);
        qcMapped.z(q[7], ca[4]);
        qcMapped.z(q[7], ca[5]);
        qcMapped.z(q[8], ca[5]);

        //z errors = indirectly via controlled x/CNOT
        qcMapped.x(q[0], ca[6]);
        qcMapped.x(q[1], ca[6]);
        qcMapped.x(q[2], ca[6]);
        qcMapped.x(q[3], ca[6]);
        qcMapped.x(q[4], ca[6]);
        qcMapped.x(q[5], ca[6]);

        qcMapped.x(q[3], ca[7]);
        qcMapped.x(q[4], ca[7]);
        qcMapped.x(q[5], ca[7]);
        qcMapped.x(q[6], ca[7]);
        qcMapped.x(q[7], ca[7]);
        qcMapped.x(q[8], ca[7]);

        for(int j=0;j<8;j++) {qcMapped.h(a[j]);}

        //MEASURE ancilla qubits
        for(int j=0;j<8;j++) {
            qcMapped.measure(a[j], clStart+j);
        }

        //CORRECT
        //x, i.e. bit flip errors
        writeClassicalControl(clStart, 1, qc::X, i);
        writeClassicalControl(clStart, 2, qc::X, i+2*nQubits);
        writeClassicalControl(clStart, 3, qc::X, i+nQubits);

        writeClassicalControl(clStart+2, 1, qc::X, i+3*nQubits);
        writeClassicalControl(clStart+2, 2, qc::X, i+5*nQubits);
        writeClassicalControl(clStart+2, 3, qc::X, i+4*nQubits);

        writeClassicalControl(clStart+4, 1, qc::X, i+6*nQubits);
        writeClassicalControl(clStart+4, 2, qc::X, i+8*nQubits);
        writeClassicalControl(clStart+4, 3, qc::X, i+7*nQubits);

        //z, i.e. phase flip errors
        writeClassicalControl(clStart+6, 1, qc::Z, i);
        writeClassicalControl(clStart+6, 2, qc::Z, i+6*nQubits);
        writeClassicalControl(clStart+6, 3, qc::Z, i+3*nQubits);
    }
}

void Q9ShorEcc::writeClassicalControl(int control, unsigned int value, qc::OpType optype, int target) {
    std::unique_ptr<qc::Operation> op = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(), dd::Qubit(target), optype);
    const auto pair_ = std::make_pair(dd::Qubit(control), dd::QubitCount(2));
    qcMapped.emplace_back<qc::ClassicControlledOperation>(op, pair_, value);
}

void Q9ShorEcc::writeDecoding() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        dd::Control ci[9];
        for(int j=0;j<9;j++) {
            ci[j] = dd::Control{dd::Qubit(i+j*nQubits), dd::Control::Type::pos};
        }

        qcMapped.x(i+nQubits, ci[0]);
        qcMapped.x(i+2*nQubits, ci[0]);

        qcMapped.x(i+4*nQubits, ci[3]);
        qcMapped.x(i+5*nQubits, ci[3]);

        qcMapped.x(i+7*nQubits, ci[6]);
        qcMapped.x(i+8*nQubits, ci[6]);

        writeToffoli(i, i+nQubits, true, i+2*nQubits, true);
        writeToffoli(i+3*nQubits, i+4*nQubits, true, i+5*nQubits, true);
        writeToffoli(i+6*nQubits, i+7*nQubits, true, i+8*nQubits, true);

        qcMapped.h(i);
        qcMapped.h(i+3*nQubits);
        qcMapped.h(i+6*nQubits);

        qcMapped.x(i+3*nQubits, ci[0]);
        qcMapped.x(i+6*nQubits, ci[0]);
        writeToffoli(i, i+3*nQubits, true, i+6*nQubits, true);
    }
    decodingDone = true;
}

void Q9ShorEcc::mapGate(const std::unique_ptr<qc::Operation> &gate) {
    if(decodingDone && gate.get()->getType()!=qc::Measure && gate.get()->getType()!=qc::H) {
        writeEncoding();
    }
    const int nQubits = qc.getNqubits();
    qc::NonUnitaryOperation *measureGate=nullptr;
    int i;
    auto type = qc::I;
    switch(gate.get()->getType()) {
    case qc::I: break;
    case qc::X:
        type = qc::Z; break;
    /*case qc::H:
        type = qc::H; break;*/
    case qc::Y:
        type = qc::Y; break;
    case qc::Z:
        type = qc::X; break;
    case qc::Measure:
        if(!decodingDone) {
            measureAndCorrect();
            writeDecoding();
        }
        measureGate = (qc::NonUnitaryOperation*)gate.get();
        for(std::size_t j=0;j<measureGate->getNclassics();j++) {
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
    for(std::size_t t=0;t<gate.get()->getNtargets();t++) {
        i = gate.get()->getTargets()[t];
        if(gate.get()->getNcontrols()==2 && decomposeMultiControlledGates) {
            //Q9Shor code: put H gate before and after each control point, i.e. "cx 0,1" becomes "h0; cz 0,1; h0"
            auto& ctrls = gate.get()->getControls();
            int idx=0;
            int ctrl2[2] = {-1, -1};
            bool ctrl2T[2] = {true, true};
            for(const auto &ct: ctrls) {
                ctrl2[idx] = ct.qubit;
                ctrl2T[idx] = ct.type == dd::Control::Type::pos;
                idx++;
            }
            for(int j=0;j<9;j++) {
                qcMapped.h(ctrl2[0]+j*nQubits);
                qcMapped.h(ctrl2[1]+j*nQubits);
            }
            if(type==qc::X) {
                for(int j=0;j<9;j++) {
                    writeToffoli(i+j*nQubits, ctrl2[0]+j*nQubits, ctrl2T[0], ctrl2[1]+j*nQubits, ctrl2T[1]);
                }
            } else if(type==qc::Z) {
                for(int j=0;j<9;j++) {
                    qcMapped.h(i+j*nQubits);
                    writeToffoli(i+j*nQubits, ctrl2[0]+j*nQubits, ctrl2T[0], ctrl2[1]+j*nQubits, ctrl2T[1]);
                    qcMapped.h(i+j*nQubits);
                }
            } else if(type==qc::Y) {
                for(int j=0;j<9;j++) {
                    writeToffoli(i+j*nQubits, ctrl2[0]+j*nQubits, ctrl2T[0], ctrl2[1]+j*nQubits, ctrl2T[1]);
                    qcMapped.h(i+j*nQubits);
                    writeToffoli(i+j*nQubits, ctrl2[0]+j*nQubits, ctrl2T[0], ctrl2[1]+j*nQubits, ctrl2T[1]);
                    qcMapped.h(i+j*nQubits);
                }
            } else {
                gateNotAvailableError(gate);
            }
            for(int j=0;j<9;j++) {
                qcMapped.h(ctrl2[0]+j*nQubits);
                qcMapped.h(ctrl2[1]+j*nQubits);
            }

        } else if(gate.get()->getNcontrols()>2 && decomposeMultiControlledGates) {
            gateNotAvailableError(gate);
        } else if(gate.get()->getNcontrols()) {
            //Q9Shor code: put H gate before and after each control point, i.e. "cx 0,1" becomes "h0; cz 0,1; h0"
            auto& ctrls = gate.get()->getControls();
            for(int j=0;j<9;j++) {
                dd::Controls ctrls2;
                for(const auto &ct: ctrls) {
                    ctrls2.insert(dd::Control{dd::Qubit(ct.qubit+j*nQubits), ct.type});
                    qcMapped.h(ct.qubit+j*nQubits);
                }
                qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls2, i+j*nQubits, type);
                for(const auto &ct: ctrls) {
                    qcMapped.h(ct.qubit+j*nQubits);
                }
            }
        } else {
            for(int j=0;j<9;j++) {
                qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i+j*nQubits, type);
            }
        }
    }
}
