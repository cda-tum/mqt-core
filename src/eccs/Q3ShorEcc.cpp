/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q3ShorEcc.hpp"

#include <chrono>
//#include <stdlib.h>

//3 data qubits, 2 for measuring -> 5 qubits per physical qubit
Q3ShorEcc::Q3ShorEcc(qc::QuantumComputation& qc): Ecc({EccID::Q3Shor, 5, 2, Q3ShorEcc::getEccName()}, qc) {}

void Q3ShorEcc::writeEccEncoding() {
	const int nQubits = qc.getNqubits();

    for(int i=0;i<nQubits;i++) {
        writeCnot(i, i+nQubits);
        writeCnot(i, i+2*nQubits);
    }
}

void Q3ShorEcc::measureAndCorrect() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {

        qcMapped.h(i+3*nQubits);
        qcMapped.h(i+4*nQubits);
        auto c3 = createControl(i+3*nQubits, true);
        auto c4 = createControl(i+4*nQubits, true);
        qcMapped.z(i,           c3);
        qcMapped.z(i+nQubits,   c3);
        qcMapped.z(i+nQubits,   c4);
        qcMapped.z(i+2*nQubits, c4);
        qcMapped.h(i+3*nQubits);
        qcMapped.h(i+4*nQubits);

        qcMapped.measure(i+3*nQubits, i);
        qcMapped.measure(i+4*nQubits, i+nQubits);

        dd::Control cn3 = createControl(i+3*nQubits, false);
        dd::Control cn4 = createControl(i+4*nQubits, false);
        dd::Controls cp3n4;
        cp3n4.insert(c3);cp3n4.insert(cn4);
        dd::Controls cp3p4;
        cp3p4.insert(c3);cp3p4.insert(c4);
        dd::Controls cn3p4;
        cn3p4.insert(cn3);cn3p4.insert(c4);

        qcMapped.x(i, cp3n4);
        qcMapped.x(i+nQubits, cp3p4);
        qcMapped.x(i+2*nQubits, cn3p4);
    }
}

void Q3ShorEcc::writeEccDecoding() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        writeCnot(i, i+nQubits);
        writeCnot(i, i+2*nQubits);
        writeToffoli(i+nQubits, i+2*nQubits, i);
    }
}

void Q3ShorEcc::mapGate(std::unique_ptr<qc::Operation> &gate) {
    const int nQubits = qc.getNqubits();
    int i;
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
        for(std::size_t j=0;j<gate.get()->getNtargets();j++) {
            i = gate.get()->getTargets()[j];
            if(gate.get()->getNcontrols()) {
                auto& ctrls = gate.get()->getControls();
                qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls, i, gate.get()->getType());
                dd::Controls ctrls2, ctrls3;
                for(const auto &ct: ctrls) {
                    ctrls2.insert(createControl(ct.qubit+nQubits, ct.type==dd::Control::Type::pos));
                    ctrls3.insert(createControl(ct.qubit+2*nQubits, ct.type==dd::Control::Type::pos));
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
        statistics.nOutputGates = -1;
        statistics.nOutputQubits = -1;
        throw qc::QFRException("Gate not possible to encode in error code!");
    }
}
