/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q9SurfaceEcc.hpp"

//TODO parameters q and c:
/*
 * q = #qubits
 * c = #classical bits
 * Assume your ECC needs p physical qubits to encode 1 logical qubit, a ancilla qubits and m measurements.
 * >>then q = p+a and c=m.
 */
Q9SurfaceEcc::Q9SurfaceEcc(qc::QuantumComputation& qc, int measureFq): Ecc({ID::Q9Surface, 17, 8, Q9SurfaceEcc::getName()}, qc, measureFq) {}

void Q9SurfaceEcc::writeEncoding() {
    measureAndCorrect();
}

void Q9SurfaceEcc::measureAndCorrect() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        unsigned int q[9];//qubits
        unsigned int a[8];//ancilla qubits
        dd::Control ca[8];//ancilla controls
        dd::Control cq[9];//qubit controls
        unsigned int m[8];
        for(int j=0;j<9;j++) { q[j] = i+j*nQubits;}
        for(int j=0;j<8;j++) { a[j] = i+(j+9)*nQubits; m[j] = i+j*nQubits; qcMapped.reset(a[j]);}
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

        for(int j=0;j<8;j++) {
            qcMapped.measure(a[j], m[j]);
        }

        //TODO logic


    }
}

void Q9SurfaceEcc::writeDecoding() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        //measure 0, 4, 8. state = m0*m4*m8
        qcMapped.measure(i, i);
        qcMapped.measure(i+4*nQubits, i+nQubits);
        qcMapped.measure(i+8*nQubits, i+2*nQubits);
        qcMapped.x(i, dd::Control{i+4*nQubits, dd::Control::Type::pos});
        qcMapped.x(i, dd::Control{i+8*nQubits, dd::Control::Type::pos});
        qcMapped.measure(i, i);
    }
}

void Q9SurfaceEcc::mapGate(const std::unique_ptr<qc::Operation> &gate) {

    const int nQubits = qc.getNqubits();
    int i;

    //currently, no control gates are supported
    if(gate.get()->getNcontrols()) {
        gateNotAvailableError(gate);
    }


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

    //for error cases, use the following line
    gateNotAvailableError(gate);
}
