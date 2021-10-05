/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q7SteaneEcc.hpp"

//7 data qubits, 6 for measuring -> 13 qubits per physical qubit
Q7SteaneEcc::Q7SteaneEcc(qc::QuantumComputation& qc): Ecc({ID::Q3Shor, 13, 6, Q7SteaneEcc::getName()}, qc) {}

void Q7SteaneEcc::writeEncoding() {
const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
    qcMapped.reset(7);
    qcMapped.reset(8);
    qcMapped.reset(9);
    qcMapped.h(7);
    qcMapped.h(8);
    qcMapped.h(9);

    auto c0 = dd::Control{dd::Qubit(7), dd::Control::Type::pos};
    auto c1 = dd::Control{dd::Qubit(8), dd::Control::Type::pos};
    auto c2 = dd::Control{dd::Qubit(9), dd::Control::Type::pos};
    auto cn0 = dd::Control{dd::Qubit(7), dd::Control::Type::neg};
    auto cn1 = dd::Control{dd::Qubit(8), dd::Control::Type::neg};
    auto cn2 = dd::Control{dd::Qubit(9), dd::Control::Type::neg};

    //K1: IIIXXXX
    qcMapped.x(3, c0);
    qcMapped.x(4, c0);
    qcMapped.x(5, c0);
    qcMapped.x(6, c0);

    //K2: XIXIXIX
    qcMapped.x(0, c1);
    qcMapped.x(2, c1);
    qcMapped.x(4, c1);
    qcMapped.x(6, c1);

    //K3: IXXIIXX
    qcMapped.x(1, c2);
    qcMapped.x(2, c2);
    qcMapped.x(5, c2);
    qcMapped.x(6, c2);

    qcMapped.h(7);
    qcMapped.h(8);
    qcMapped.h(9);

    qcMapped.measure(7, 0);
    qcMapped.measure(8, 1);
    qcMapped.measure(9, 2);

    //correct Z_i for i+1 = c1*1+c2*2+c0*4
    qcMapped.z(0, {c1, cn2, cn0});
    qcMapped.z(1, {cn1, c2, c0});
    qcMapped.z(2, {c1, c2, cn0});
    qcMapped.z(3, {cn1, cn2, c0});
    qcMapped.z(4, {c1, cn2, c0});
    qcMapped.z(5, {cn1, c2, c0});
    qcMapped.z(6, {c1, c2, c0});
    }
}

void Q7SteaneEcc::measureAndCorrect() {
    writeEncoding();//identical

    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
    qcMapped.reset(10);
    qcMapped.reset(11);
    qcMapped.reset(12);
    qcMapped.h(10);
    qcMapped.h(11);
    qcMapped.h(12);

    auto c0 = dd::Control{dd::Qubit(10), dd::Control::Type::pos};
    auto c1 = dd::Control{dd::Qubit(11), dd::Control::Type::pos};
    auto c2 = dd::Control{dd::Qubit(12), dd::Control::Type::pos};
    auto cn0 = dd::Control{dd::Qubit(10), dd::Control::Type::neg};
    auto cn1 = dd::Control{dd::Qubit(11), dd::Control::Type::neg};
    auto cn2 = dd::Control{dd::Qubit(12), dd::Control::Type::neg};

    //K1: IIIXXXX
    qcMapped.z(3, c0);
    qcMapped.z(4, c0);
    qcMapped.z(5, c0);
    qcMapped.z(6, c0);

    //K2: XIXIXIX
    qcMapped.z(0, c1);
    qcMapped.z(2, c1);
    qcMapped.z(4, c1);
    qcMapped.z(6, c1);

    //K3: IXXIIXX
    qcMapped.z(1, c2);
    qcMapped.z(2, c2);
    qcMapped.z(5, c2);
    qcMapped.z(6, c2);

    qcMapped.h(10);
    qcMapped.h(11);
    qcMapped.h(12);

    qcMapped.measure(10, 3);
    qcMapped.measure(11, 4);
    qcMapped.measure(12, 5);

    //correct X_i for i+1 = c1*1+c2*2+c0*4
    qcMapped.x(0, {c1, cn2, cn0});
    qcMapped.x(1, {cn1, c2, c0});
    qcMapped.x(2, {c1, c2, cn0});
    qcMapped.x(3, {cn1, cn2, c0});
    qcMapped.x(4, {c1, cn2, c0});
    qcMapped.x(5, {cn1, c2, c0});
    qcMapped.x(6, {c1, c2, c0});
    }
}

void Q7SteaneEcc::writeDecoding() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        //TODO
    }
}

void Q7SteaneEcc::mapGate(const std::unique_ptr<qc::Operation> &gate) {
    const int nQubits = qc.getNqubits();
    switch(gate.get()->getType()) {
    case qc::I: break;
    case qc::X:
    case qc::H:
    case qc::Y:
    case qc::Z:
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
