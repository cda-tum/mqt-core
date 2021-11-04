/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q7SteaneEcc.hpp"

//7 data qubits, 6 for measuring -> 13 qubits per physical qubit (7 classical for measuring at end)
Q7SteaneEcc::Q7SteaneEcc(qc::QuantumComputation& qc, int measureFq): Ecc({ID::Q3Shor, 13, 7, Q7SteaneEcc::getName()}, qc, measureFq) {}

void Q7SteaneEcc::writeEncoding() {
const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
    qcMapped.reset(i+7*nQubits);
    qcMapped.reset(i+8*nQubits);
    qcMapped.reset(i+9*nQubits);
    qcMapped.h(i+7*nQubits);
    qcMapped.h(i+8*nQubits);
    qcMapped.h(i+9*nQubits);

    auto c0 = dd::Control{dd::Qubit(i+nQubits*7), dd::Control::Type::pos};
    auto c1 = dd::Control{dd::Qubit(i+nQubits*8), dd::Control::Type::pos};
    auto c2 = dd::Control{dd::Qubit(i+nQubits*9), dd::Control::Type::pos};
    auto cn0 = dd::Control{dd::Qubit(i+nQubits*7), dd::Control::Type::neg};
    auto cn1 = dd::Control{dd::Qubit(i+nQubits*8), dd::Control::Type::neg};
    auto cn2 = dd::Control{dd::Qubit(i+nQubits*9), dd::Control::Type::neg};

    //K1: IIIXXXX
    qcMapped.x(i+nQubits*3, c0);
    qcMapped.x(i+nQubits*4, c0);
    qcMapped.x(i+nQubits*5, c0);
    qcMapped.x(i+nQubits*6, c0);

    //K2: XIXIXIX
    qcMapped.x(i+nQubits*0, c1);
    qcMapped.x(i+nQubits*2, c1);
    qcMapped.x(i+nQubits*4, c1);
    qcMapped.x(i+nQubits*6, c1);

    //K3: IXXIIXX
    qcMapped.x(i+nQubits*1, c2);
    qcMapped.x(i+nQubits*2, c2);
    qcMapped.x(i+nQubits*5, c2);
    qcMapped.x(i+nQubits*6, c2);

    qcMapped.h(i+nQubits*7);
    qcMapped.h(i+nQubits*8);
    qcMapped.h(i+nQubits*9);

    qcMapped.measure(i+nQubits*7, i+nQubits*0);
    qcMapped.measure(i+nQubits*8, i+nQubits*1);
    qcMapped.measure(i+nQubits*9, i+nQubits*2);

    //correct Z_i for i+1 = c1*1+c2*2+c0*4
    qcMapped.z(i+nQubits*0, {c1, cn2, cn0});
    qcMapped.z(i+nQubits*1, {cn1, c2, c0});
    qcMapped.z(i+nQubits*2, {c1, c2, cn0});
    qcMapped.z(i+nQubits*3, {cn1, cn2, c0});
    qcMapped.z(i+nQubits*4, {c1, cn2, c0});
    qcMapped.z(i+nQubits*5, {cn1, c2, c0});
    qcMapped.z(i+nQubits*6, {c1, c2, c0});
    }
}

void Q7SteaneEcc::measureAndCorrect() {
    writeEncoding();//identical

    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        qcMapped.reset(i+nQubits*10);
        qcMapped.reset(i+nQubits*11);
        qcMapped.reset(i+nQubits*12);
        qcMapped.h(i+nQubits*10);
        qcMapped.h(i+nQubits*11);
        qcMapped.h(i+nQubits*12);

        auto c0 = dd::Control{dd::Qubit(i+nQubits*10), dd::Control::Type::pos};
        auto c1 = dd::Control{dd::Qubit(i+nQubits*11), dd::Control::Type::pos};
        auto c2 = dd::Control{dd::Qubit(i+nQubits*12), dd::Control::Type::pos};
        auto cn0 = dd::Control{dd::Qubit(i+nQubits*10), dd::Control::Type::neg};
        auto cn1 = dd::Control{dd::Qubit(i+nQubits*11), dd::Control::Type::neg};
        auto cn2 = dd::Control{dd::Qubit(i+nQubits*12), dd::Control::Type::neg};

    //K1: IIIXXXX
    qcMapped.z(i+nQubits*3, c0);
    qcMapped.z(i+nQubits*4, c0);
    qcMapped.z(i+nQubits*5, c0);
    qcMapped.z(i+nQubits*6, c0);

    //K2: XIXIXIX
    qcMapped.z(i+nQubits*0, c1);
    qcMapped.z(i+nQubits*2, c1);
    qcMapped.z(i+nQubits*4, c1);
    qcMapped.z(i+nQubits*6, c1);

    //K3: IXXIIXX
    qcMapped.z(i+nQubits*1, c2);
    qcMapped.z(i+nQubits*2, c2);
    qcMapped.z(i+nQubits*5, c2);
    qcMapped.z(i+nQubits*6, c2);

    qcMapped.h(i+nQubits*10);
    qcMapped.h(i+nQubits*11);
    qcMapped.h(i+nQubits*12);

    qcMapped.measure(i+nQubits*10, i+nQubits*3);
    qcMapped.measure(i+nQubits*11, i+nQubits*4);
    qcMapped.measure(i+nQubits*12, i+nQubits*5);

        //correct X_i for i+1 = c1*1+c2*2+c0*4
        qcMapped.x(i+nQubits*0, {c1, cn2, cn0});
        qcMapped.x(i+nQubits*1, {cn1, c2, c0});
        qcMapped.x(i+nQubits*2, {c1, c2, cn0});
        qcMapped.x(i+nQubits*3, {cn1, cn2, c0});
        qcMapped.x(i+nQubits*4, {c1, cn2, c0});
        qcMapped.x(i+nQubits*5, {cn1, c2, c0});
        qcMapped.x(i+nQubits*6, {c1, c2, c0});
        }
}

void Q7SteaneEcc::writeDecoding() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        for(int j=0;j<7;j++) {
            qcMapped.measure(i+nQubits*j, i+nQubits*j);
        }
        //odd number of 1's = 1, even number of 1's = 0 --> odd number of cnot result in 1, even number result in 0 (on target)
        for(int j=1;j<7;j++) {
            auto c = dd::Control{dd::Qubit(i+nQubits*j), dd::Control::Type::pos};
            qcMapped.x(i, c);
        }
        qcMapped.measure(i, i);
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
        for(std::size_t t=0;t<gate.get()->getNtargets();t++) {
            int i = gate.get()->getTargets()[t];
            if(gate.get()->getNcontrols()) {
                auto& ctrls = gate.get()->getControls();
                for(int j=0;j<7;j++) {
                    dd::Controls ctrls2;
                    for(const auto &ct: ctrls) {
                        ctrls2.insert(dd::Control{dd::Qubit(ct.qubit+j*nQubits), ct.type});
                    }
                    qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls2, i+j*nQubits, gate.get()->getType());
                }
            } else {
                for(int j=0;j<7;j++) {
                    qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i+j*nQubits, gate.get()->getType());
                }
            }
        }
        break;
    //locigal S = 3 physical S's
    case qc::S:
    case qc::Sdag:
        for(std::size_t t=0;t<gate.get()->getNtargets();t++) {
            int i = gate.get()->getTargets()[t];
            if(gate.get()->getNcontrols()) {
                auto& ctrls = gate.get()->getControls();
                for(int j=0;j<7;j++) {
                    dd::Controls ctrls2;
                    for(const auto &ct: ctrls) {
                        ctrls2.insert(dd::Control{dd::Qubit(ct.qubit+j*nQubits), ct.type});
                    }
                    qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls2, i+j*nQubits, gate.get()->getType());
                    qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls2, i+j*nQubits, gate.get()->getType());
                    qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls2, i+j*nQubits, gate.get()->getType());
                }
            } else {
                for(int j=0;j<7;j++) {
                    qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i+j*nQubits, gate.get()->getType());
                    qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i+j*nQubits, gate.get()->getType());
                    qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i+j*nQubits, gate.get()->getType());
                }
            }
        }
        break;
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
    case qc::Measure:
    break;
    }
}

