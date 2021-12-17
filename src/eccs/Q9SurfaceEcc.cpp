/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q9SurfaceEcc.hpp"

//This code has been described in https://arxiv.org/pdf/1608.05053.pdf
Q9SurfaceEcc::Q9SurfaceEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly): Ecc({ID::Q9Surface, 9, 8, Q9SurfaceEcc::getName()}, qc, measureFq, decomposeMC, cliffOnly) {}

void Q9SurfaceEcc::initMappedCircuit() {
//method is overridden because we need 2 kinds of classical measurement output registers
    qc.stripIdleQubits(true, false);
    statistics.nInputQubits = qc.getNqubits();
    statistics.nInputClassicalBits = (int)qc.getNcbits();
	statistics.nOutputQubits = qc.getNqubits()*ecc.nRedundantQubits+ecc.nCorrectingBits;
	statistics.nOutputClassicalBits = statistics.nInputClassicalBits+ecc.nCorrectingBits;
	qcMapped.addQubitRegister(statistics.nOutputQubits);
	qcMapped.addClassicalRegister(statistics.nInputClassicalBits);
	qcMapped.addClassicalRegister(4, "qeccX");
	qcMapped.addClassicalRegister(4, "qeccZ");
}

void Q9SurfaceEcc::writeEncoding() {
    measureAndCorrect();
    decodingDone = false;
}

void Q9SurfaceEcc::measureAndCorrect() {
    const int nQubits = qc.getNqubits();
    const int ancStart = qc.getNqubits()*ecc.nRedundantQubits;
    const int clAncStart = qc.getNcbits();
    for(int i=0;i<nQubits;i++) {
        unsigned int q[9];//qubits
        unsigned int a[8];//ancilla qubits
        dd::Control ca[8];//ancilla controls
        dd::Control cq[9];//qubit controls
        for(int j=0;j<9;j++) { q[j] = i+j*nQubits;}
        for(int j=0;j<8;j++) { a[j] = ancStart+j; qcMapped.reset(a[j]);}
        for(int j=0;j<8;j++) { ca[j] = dd::Control{dd::Qubit(a[j]), dd::Control::Type::pos}; }
        for(int j=0;j<9;j++) { cq[j] = dd::Control{dd::Qubit(q[j]), dd::Control::Type::pos}; }

        //X-type check on a0, a2, a5, a7: cx a->q
        //Z-type check on a1, a3, a4, a6: cz a->q = cx q->a, no hadamard gate
        qcMapped.h(a[0]);
        qcMapped.h(a[2]);
        qcMapped.h(a[5]);
        qcMapped.h(a[7]);

        qcMapped.x(a[6], cq[8]);
        qcMapped.x(q[7], ca[5]);
        qcMapped.x(a[4], cq[6]);
        qcMapped.x(a[3], cq[4]);
        qcMapped.x(q[3], ca[2]);
        qcMapped.x(q[1], ca[0]);

        qcMapped.x(a[6], cq[5]);
        qcMapped.x(a[4], cq[3]);
        qcMapped.x(q[8], ca[5]);
        qcMapped.x(a[3], cq[1]);
        qcMapped.x(q[4], ca[2]);
        qcMapped.x(q[2], ca[0]);

        qcMapped.x(q[6], ca[7]);
        qcMapped.x(q[4], ca[5]);
        qcMapped.x(a[4], cq[7]);
        qcMapped.x(a[3], cq[5]);
        qcMapped.x(q[0], ca[2]);
        qcMapped.x(a[1], cq[3]);

        qcMapped.x(q[7], ca[7]);
        qcMapped.x(q[5], ca[5]);
        qcMapped.x(a[4], cq[4]);
        qcMapped.x(a[3], cq[2]);
        qcMapped.x(q[1], ca[2]);
        qcMapped.x(a[1], cq[0]);


        qcMapped.h(a[0]);
        qcMapped.h(a[2]);
        qcMapped.h(a[5]);
        qcMapped.h(a[7]);

        qcMapped.measure(a[0], clAncStart);
        qcMapped.measure(a[2], clAncStart+1);
        qcMapped.measure(a[5], clAncStart+2);
        qcMapped.measure(a[7], clAncStart+3);

        qcMapped.measure(a[1], clAncStart+4);
        qcMapped.measure(a[3], clAncStart+5);
        qcMapped.measure(a[4], clAncStart+6);
        qcMapped.measure(a[6], clAncStart+7);

        //logic
        writeClassicalControl(clAncStart, 1, qc::Z, q[2]); //a[0]
        writeClassicalControl(clAncStart, 2, qc::Z, q[3]); //a[2] (or q[0])
        writeClassicalControl(clAncStart, 3, qc::Z, q[1]); //a[0,2]
        writeClassicalControl(clAncStart, 4, qc::Z, q[5]); //a[5] (or q[8])
        writeClassicalControl(clAncStart, 6, qc::Z, q[4]); //a[2,5]
        writeClassicalControl(clAncStart, 8, qc::Z, q[6]); //a[7]
        writeClassicalControl(clAncStart, 12, qc::Z, q[7]); //a[5,7]

        writeClassicalControl(clAncStart+4, 1, qc::X, q[0]); //a[1]
        writeClassicalControl(clAncStart+4, 2, qc::X, q[1]); //a[3] (or q[2])
        writeClassicalControl(clAncStart+4, 4, qc::X, q[7]); //a[4] (or q[6])
        writeClassicalControl(clAncStart+4, 5, qc::X, q[3]); //a[1,4]
        writeClassicalControl(clAncStart+4, 6, qc::X, q[4]); //a[3,4]
        writeClassicalControl(clAncStart+4, 8, qc::X, q[8]); //a[6]
        writeClassicalControl(clAncStart+4, 10, qc::X, q[5]); //a[3,6]


    }
}

void Q9SurfaceEcc::writeDecoding() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        //measure 0, 4, 8. state = m0*m4*m8
        qcMapped.measure(i, i);
        qcMapped.measure(i+4*nQubits, i);
        qcMapped.measure(i+8*nQubits, i);
        qcMapped.x(i, dd::Control{i+4*nQubits, dd::Control::Type::pos});
        qcMapped.x(i, dd::Control{i+8*nQubits, dd::Control::Type::pos});
        qcMapped.measure(i, i);
    }
    decodingDone = true;
}

void Q9SurfaceEcc::writeClassicalControl(int control, unsigned int value, qc::OpType optype, int target) {
    std::unique_ptr<qc::Operation> op = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(), dd::Qubit(target), optype);
    const auto pair_ = std::make_pair(dd::Qubit(control), dd::QubitCount(4));
    qcMapped.emplace_back<qc::ClassicControlledOperation>(op, pair_, value);
}

void Q9SurfaceEcc::mapGate(const std::unique_ptr<qc::Operation> &gate) {
    if(decodingDone && gate.get()->getType()!=qc::Measure) {
        writeEncoding();
    }
    const int nQubits = qc.getNqubits();
    int i;

    //currently, no control gates are supported
    if(gate.get()->getNcontrols() && gate.get()->getType() != qc::Measure) {
        gateNotAvailableError(gate);
    }

    qc::NonUnitaryOperation *measureGate=nullptr;
    switch(gate.get()->getType()) {
    case qc::I: break;
    case qc::X:
        for(std::size_t t=0;t<gate.get()->getNtargets();t++) {
            i = gate.get()->getTargets()[t];
            qcMapped.x(i+2*nQubits);
            qcMapped.x(i+4*nQubits);
            qcMapped.x(i+6*nQubits);
        }
        break;
    case qc::H:
        for(std::size_t t=0;t<gate.get()->getNtargets();t++) {
            i = gate.get()->getTargets()[t];
            qcMapped.h(i);
            qcMapped.h(i+2*nQubits);
            qcMapped.h(i+4*nQubits);
            qcMapped.h(i+6*nQubits);
            qcMapped.h(i+8*nQubits);
        }
        break;
    case qc::Y:
        //Y = Z X
        for(std::size_t t=0;t<gate.get()->getNtargets();t++) {
            i = gate.get()->getTargets()[t];
            qcMapped.z(i);
            qcMapped.z(i+4*nQubits);
            qcMapped.z(i+8*nQubits);
            qcMapped.x(i+2*nQubits);
            qcMapped.x(i+4*nQubits);
            qcMapped.x(i+6*nQubits);
        }
        break;
    case qc::Z:
        for(std::size_t t=0;t<gate.get()->getNtargets();t++) {
            i = gate.get()->getTargets()[t];
            qcMapped.z(i);
            qcMapped.z(i+4*nQubits);
            qcMapped.z(i+8*nQubits);
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

    }
}
