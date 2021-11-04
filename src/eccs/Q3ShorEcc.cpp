/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q3ShorEcc.hpp"

//3 data qubits, 2 for measuring -> 5 qubits per physical qubit
Q3ShorEcc::Q3ShorEcc(qc::QuantumComputation& qc, int measureFq): Ecc({ID::Q3Shor, 5, 2, Q3ShorEcc::getName()}, qc, measureFq) {}

void Q3ShorEcc::writeEncoding() {
	const int nQubits = qc.getNqubits();

    for(int i=0;i<nQubits;i++) {
        auto ctrl = dd::Control{dd::Qubit(i), dd::Control::Type::pos};
        qcMapped.x(i+nQubits, ctrl);
        qcMapped.x(i+2*nQubits, ctrl);
    }
}

void Q3ShorEcc::measureAndCorrect() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {

        qcMapped.h(i+3*nQubits);
        qcMapped.h(i+4*nQubits);
        auto c3 = dd::Control{dd::Qubit(i+3*nQubits), dd::Control::Type::pos};
        auto c4 = dd::Control{dd::Qubit(i+4*nQubits), dd::Control::Type::pos};
        qcMapped.z(i,           c3);
        qcMapped.z(i+nQubits,   c3);
        qcMapped.z(i+nQubits,   c4);
        qcMapped.z(i+2*nQubits, c4);
        qcMapped.h(i+3*nQubits);
        qcMapped.h(i+4*nQubits);

        qcMapped.measure(i+3*nQubits, i);
        qcMapped.measure(i+4*nQubits, i+nQubits);

        auto cn3 = dd::Control{dd::Qubit(i+3*nQubits), dd::Control::Type::neg};
        auto cn4 = dd::Control{dd::Qubit(i+4*nQubits), dd::Control::Type::neg};

        qcMapped.x(i, {c3, cn4});
        qcMapped.x(i+nQubits, {c3, c4});
        qcMapped.x(i+2*nQubits, {cn3, c4});
    }
}

void Q3ShorEcc::writeDecoding() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        auto ctrl = dd::Control{dd::Qubit(i), dd::Control::Type::pos};
        qcMapped.x(i+nQubits, ctrl);
        qcMapped.x(i+2*nQubits, ctrl);

        dd::Controls ctrls;
        ctrls.insert(dd::Control{dd::Qubit(i+nQubits), dd::Control::Type::pos});
        ctrls.insert(dd::Control{dd::Qubit(i+2*nQubits)});
        qcMapped.x(i, ctrls);
    }
}

void Q3ShorEcc::mapGate(const std::unique_ptr<qc::Operation> &gate) {
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
        for(std::size_t j=0;j<gate.get()->getNtargets();j++) {
            auto i = gate.get()->getTargets()[j];
            if(gate.get()->getNcontrols()) {
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
