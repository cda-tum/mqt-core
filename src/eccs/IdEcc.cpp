/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/IdEcc.hpp"

IdEcc::IdEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly):
    Ecc({ID::Id, 1, 0, IdEcc::getName()}, qc, measureFq, decomposeMC, cliffOnly) {
    this->qcMapped = qc.clone();
}

void IdEcc::writeEncoding() {}

void IdEcc::measureAndCorrect() {}

void IdEcc::writeDecoding() {}

void IdEcc::mapGate(const std::unique_ptr<qc::Operation>& gate, qc::QuantumComputation& qc) {
    //gates have already been written to 'qcMapped' in the constructor
    if (gate->getNcontrols() > 1 && decomposeMultiControlledGates) {
        gateNotAvailableError(gate);
    }
    if (cliffordGatesOnly) {
        switch (gate->getType()) {
            case qc::H:
            case qc::X:
            case qc::Y:
            case qc::Z:
            case qc::I:
            case qc::S:
            case qc::T:
                break;
            default:
                gateNotAvailableError(gate);
                break;
        }
    }
}
