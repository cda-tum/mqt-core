/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/QxCustomEcc.hpp"

//TODO parameters q and c:
/*
 * q = #qubits
 * c = #classical bits
 * Assume your ECC needs p physical qubits to encode 1 logical qubit, a ancilla qubits and m measurements.
 * >>then q = p+a and c=m.
 */
QxCustomEcc::QxCustomEcc(qc::QuantumComputation& qc, int measureFq): Ecc({ID::QxCustom, /*q*/-1, /*c*/-1, QxCustomEcc::getName()}, qc, measureFq) {}

void QxCustomEcc::writeEncoding() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        //TODO write encoding here in 'encoded' form into the variable 'qcMapped'
        //i.e. which operations do you need to perform to get an "all |0>" state?

        //if you do not encode each qubit for itself, feel free to remove the loop :)
    }

}

void QxCustomEcc::measureAndCorrect() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        //TODO write correcting here in 'encoded' form into the variable 'qcMapped'
        //i.e. which operations do you need to perform to get correct a state in case of errors?

        //if you do not encode each qubit for itself, feel free to remove the loop :)
    }
}

void QxCustomEcc::writeDecoding() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        //TODO write correcting here in 'encoded' form into the variable 'qcMapped'
        //i.e. which operations do you need to perform to get the encoded information back into the original qubits?

        //if you do not encode each qubit for itself, feel free to remove the loop :)
    }
}

void QxCustomEcc::mapGate(const std::unique_ptr<qc::Operation> &gate) {

    //TODO make sure the parameter gate is written in 'encoded' form into the variable 'qcMapped'

    //for error cases, use the following line
    gateNotAvailableError(gate);
}
