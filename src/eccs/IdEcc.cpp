/*
 * This file is part of the JKQ QMAP library which is released under the MIT license.
 * See file README.md or go to https://iic.jku.at/eda/research/ibm_qx_mapping/ for more information.
 */

#include "eccs/IdEcc.hpp"

#include <chrono>
//#include <stdlib.h>

IdEcc::IdEcc(qc::QuantumComputation& qc) : Ecc({EccID::Id, 1, 0, IdEcc::getEccName()}, qc) {
}


void IdEcc::writeEccEncoding() {}

void IdEcc::measureAndCorrect() {}

void IdEcc::writeEccDecoding() {}

void IdEcc::mapGate(std::unique_ptr<qc::Operation> &gate) {
    const int nQubits = qc.getNqubits();
    int i;
    switch(gate.get()->getType()) {
    case qc::I: break;
    case qc::X:
    case qc::H:
    case qc::Y:
    case qc::Z:
    //TODO check S, T, V
    case qc::S:
    case qc::Sdag:
    case qc::T:
    case qc::Tdag:
    case qc::V:
    case qc::Vdag:
        //TODO controlled/multitarget check
        i = gate.get()->getTargets()[0];
        qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i, gate.get()->getType());
        break;

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

