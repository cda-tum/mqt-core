/*
 * This file is part of the JKQ QMAP library which is released under the MIT license.
 * See file README.md or go to https://iic.jku.at/eda/research/ibm_qx_mapping/ for more information.
 */

#include "eccs/Q9ShorEcc.hpp"

#include <chrono>
//#include <stdlib.h>


Q9ShorEcc::Q9ShorEcc(qc::QuantumComputation& qc) : Ecc({EccID::Q9Shor, 9, Q9ShorEcc::getEccName()}, qc) {
}

void Q9ShorEcc::writeEccEncoding() {
	const int nQubits = qc.getNqubits();
	const int nQubitsMapped = qcMapped.getNqubits();
    for(int i=0;i<nQubits;i++) {
        writeCnot(i, i+3*nQubits);
        writeCnot(i, i+6*nQubits);
        qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, i, qc::H);
        qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, i+3*nQubits, qc::H);
        qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, i+6*nQubits, qc::H);
        writeCnot(i, i+nQubits);
        writeCnot(i, i+2*nQubits);
        writeCnot(i+3*nQubits, i+4*nQubits);
        writeCnot(i+3*nQubits, i+5*nQubits);
        writeCnot(i+6*nQubits, i+7*nQubits);
        writeCnot(i+6*nQubits, i+8*nQubits);
    }
}

void Q9ShorEcc::writeEccDecoding() {
    const int nQubits = qc.getNqubits();
    const int nQubitsMapped = qcMapped.getNqubits();
    for(int i=0;i<nQubits;i++) {
        writeCnot(i, i+nQubits);
        writeCnot(i, i+2*nQubits);
        writeToffoli(i+nQubits, i+2*nQubits, i);
        writeCnot(i+3*nQubits, i+4*nQubits);
        writeCnot(i+3*nQubits, i+5*nQubits);
        writeToffoli(i+4*nQubits, i+5*nQubits, i+3*nQubits);
        writeCnot(i+6*nQubits, i+7*nQubits);
        writeCnot(i+6*nQubits, i+8*nQubits);
        writeToffoli(i+7*nQubits, i+8*nQubits, i+6*nQubits);

        qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, i, qc::H);
        qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, i+3*nQubits, qc::H);
        qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, i+6*nQubits, qc::H);

        writeCnot(i, i+3*nQubits);
        writeCnot(i, i+6*nQubits);
        writeToffoli(i+3*nQubits, i+6*nQubits, i);
    }
}

void Q9ShorEcc::mapGate(std::unique_ptr<qc::Operation> &gate) {
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

    //TODO check S, T, V
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
        statistics.nOutputGates = -1;
        statistics.nOutputQubits = -1;
        throw qc::QFRException("Gate not possible to encode in error code!");
    }
    //TODO controlled/multitarget check
    i = gate.get()->getTargets()[0];
    for(int j=0;j<9;j++) {
        qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i+j*nQubits, type);
    }
}
