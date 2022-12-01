/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/QxCustomEcc.hpp"

void QxCustomEcc::writeEncoding() {
    const int nQubits = qcOriginal.getNqubits();
    for (int i = 0; i < nQubits; i++) {
        //TODO write encoding here in 'encoded' form into the variable 'qcMapped'
        //i.e. which operations do you need to perform to get an "all |0>" state?

        //if you do not encode each qubit for itself, feel free to remove the loop :)
    }
}

void QxCustomEcc::measureAndCorrect() {
    const int nQubits = qcOriginal.getNqubits();
    for (int i = 0; i < nQubits; i++) {
        //TODO write correcting here in 'encoded' form into the variable 'qcMapped'
        //i.e. which operations do you need to perform to get correct a state in case of errors?

        //if you do not encode each qubit for itself, feel free to remove the loop :)
    }
}

void QxCustomEcc::writeDecoding() {
    const int nQubits = qcOriginal.getNqubits();
    for (int i = 0; i < nQubits; i++) {
        //TODO write correcting here in 'encoded' form into the variable 'qcMapped'
        //i.e. which operations do you need to perform to get the encoded information back into the original qubits?

        //if you do not encode each qubit for itself, feel free to remove the loop :)
    }
}

void QxCustomEcc::mapGate(const qc::Operation& gate) {
    //TODO make sure the parameter gate is written in 'encoded' form into the variable 'qcMapped'

    //for error cases, use the following line
    gateNotAvailableError(gate);
}
