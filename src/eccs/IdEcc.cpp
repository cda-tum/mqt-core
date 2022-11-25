/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/IdEcc.hpp"

IdEcc::IdEcc(qc::QuantumComputation& qc, int measureFq):
    Ecc({ID::Id, 1, 0, IdEcc::getName()}, qc, measureFq) {}

void IdEcc::writeEncoding() {
    //Not necessary for dummy ecc
}

void IdEcc::measureAndCorrect() {
    //Not necessary for dummy ecc
}

void IdEcc::writeDecoding() {
    //Not necessary for dummy ecc
}

void IdEcc::mapGate(const qc::Operation& gate) {
    switch (gate.getType()) {
        case qc::H:
        case qc::X:
        case qc::Y:
        case qc::Z:
        case qc::I:
        case qc::S:
        case qc::T:
        case qc::Tdag:
            qcMapped.emplace_back<dd::StandardOperation>(qcOriginal.getNqubits(), gate.getControls(), gate.getTargets(), gate.getType());
            break;
        case qc::Measure:
            if (auto measureGate = dynamic_cast<const qc::NonUnitaryOperation*>(&gate)) {
                for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
                    qcMapped.measure(measureGate->getTargets()[j], measureGate->getClassics()[j]);
                }
            } else {
                throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
            }
            break;
        default:
            gateNotAvailableError(gate);
            break;
    }
}
