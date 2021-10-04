/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q9ShorEcc.hpp"

//9 data qubits, 8 for measuring -> 17 qubits per physical qubit
Q9ShorEcc::Q9ShorEcc(qc::QuantumComputation& qc) : Ecc({ID::Q9Shor, 17, 8, Q9ShorEcc::getName()}, qc) {}

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
}

void Q9ShorEcc::measureAndCorrect() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        //syntactic sugar for qubit indices
        unsigned int q[9];//qubits
        unsigned int a[8];//ancilla qubits
        dd::Control ca[8];//ancilla controls
        unsigned int m[8];
        for(int j=0;j<9;j++) { q[j] = i+j*nQubits;}
        for(int j=0;j<8;j++) { a[j] = i+(j+9)*nQubits; m[j] = i+j*nQubits;}
        for(int j=0;j<8;j++) { ca[j] = dd::Control{dd::Qubit(a[j]), dd::Control::Type::pos}; }


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
            qcMapped.measure(a[j], m[j]);
        }

        //CORRECT
        //x, i.e. bit flip errors
        dd::Controls cp0n1;
        cp0n1.insert(ca[0]);
        cp0n1.insert(dd::Control{dd::Qubit(a[1]), dd::Control::Type::neg});
        dd::Controls cp0p1;
        cp0p1.insert(ca[0]);
        cp0p1.insert(ca[1]);
        dd::Controls cn0p1;
        cn0p1.insert(dd::Control{dd::Qubit(a[0]), dd::Control::Type::neg});
        cn0p1.insert(ca[1]);
        qcMapped.x(q[0], cp0n1);
        qcMapped.x(q[1], cp0p1);
        qcMapped.x(q[2], cn0p1);

        dd::Controls cp2n3;
        cp2n3.insert(ca[2]);
        cp2n3.insert(dd::Control{dd::Qubit(a[3]), dd::Control::Type::neg});
        dd::Controls cp2p3;
        cp2p3.insert(ca[2]);
        cp2p3.insert(ca[3]);
        dd::Controls cn2p3;
        cn2p3.insert(dd::Control{dd::Qubit(a[2]), dd::Control::Type::neg});
        cn2p3.insert(ca[3]);
        qcMapped.x(q[3], cp2n3);
        qcMapped.x(q[4], cp2p3);
        qcMapped.x(q[5], cn2p3);

        dd::Controls cp4n5;
        cp4n5.insert(ca[4]);
        cp4n5.insert(dd::Control{dd::Qubit(a[5]), dd::Control::Type::neg});
        dd::Controls cp4p5;
        cp4p5.insert(ca[4]);
        cp4p5.insert(ca[5]);
        dd::Controls cn4p5;
        cn4p5.insert(dd::Control{dd::Qubit(a[4]), dd::Control::Type::neg});
        cn4p5.insert(ca[5]);
        qcMapped.x(q[6], cp4n5);
        qcMapped.x(q[7], cp4p5);
        qcMapped.x(q[8], cn4p5);


        //z, i.e. phase flip errors
        dd::Controls cp6n7;
        cp6n7.insert(ca[6]);
        cp6n7.insert(dd::Control{dd::Qubit(a[7]), dd::Control::Type::neg});
        dd::Controls cp6p7;
        cp6p7.insert(ca[6]);
        cp6p7.insert(ca[7]);
        dd::Controls cn6p7;
        cn6p7.insert(dd::Control{dd::Qubit(a[6]), dd::Control::Type::neg});
        cn6p7.insert(ca[7]);
        qcMapped.z(q[0], cp6n7);
        qcMapped.z(q[3], cp6p7);
        qcMapped.z(q[6], cn6p7);

    }
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

        dd::Controls c12;
        c12.insert(ci[1]);
        c12.insert(ci[2]);
        qcMapped.x(i, c12);

        dd::Controls c45;
        c45.insert(ci[4]);
        c45.insert(ci[5]);
        qcMapped.x(i+3*nQubits, c45);

        dd::Controls c78;
        c78.insert(ci[7]);
        c78.insert(ci[8]);
        qcMapped.x(i+6*nQubits, c78);

        qcMapped.h(i);
        qcMapped.h(i+3*nQubits);
        qcMapped.h(i+6*nQubits);

        qcMapped.x(i+3*nQubits, ci[0]);
        qcMapped.x(i+6*nQubits, ci[0]);

        dd::Controls c36;
        c36.insert(ci[3]);
        c36.insert(ci[6]);
        qcMapped.x(i, c36);
    }
}

void Q9ShorEcc::mapGate(const std::unique_ptr<qc::Operation> &gate) {
    const int nQubits = qc.getNqubits();
    int i;
    auto type = qc::I;
    switch(gate.get()->getType()) {
    case qc::I: break;
    case qc::X:
        type = qc::Z; break;
    case qc::H:
        type = qc::H; break;
    case qc::Y:
        type = qc::Y; break;
    case qc::Z:
        type = qc::X; break;

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
        if(gate.get()->getNcontrols()) {
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
