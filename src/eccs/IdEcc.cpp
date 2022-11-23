/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/IdEcc.hpp"

IdEcc::IdEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly):
    Ecc({ID::Id, 1, 0, IdEcc::getName()}, qc, measureFq, decomposeMC, cliffOnly) {}

void IdEcc::writeEncoding() {}

void IdEcc::measureAndCorrect() {}

void IdEcc::writeDecoding() {}

void IdEcc::mapGate(const std::unique_ptr<qc::Operation>& gate, [[maybe_unused]] qc::QuantumComputation& qc) {
    qc::NonUnitaryOperation* measureGate;

    //gates have already been written to 'qcMapped' in the constructor
    if (cliffordGatesOnly) {
        gateNotAvailableError(gate);
    }
    switch (gate->getType()) {
        case qc::H:
        case qc::X:
        case qc::Y:
        case qc::Z:
        case qc::I:
        case qc::S:
        case qc::T:
        case qc::Tdag:
            qcMapped.emplace_back<dd::StandardOperation>(qc.getNqubits(), gate->getControls(), gate->getTargets(), gate->getType());
            break;
        case qc::Measure:
            measureGate = (qc::NonUnitaryOperation*)gate.get();
            for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
                qcMapped.measure(measureGate->getTargets()[j], measureGate->getClassics()[j]);
            }
            break;
        default:
            gateNotAvailableError(gate);
            break;
    }
}
