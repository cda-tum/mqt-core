/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "ecc/QxCustomEcc.hpp"

void QxCustomEcc::writeEncoding() {
    for (std::size_t i = 0; i < qcOriginal->getNqubits(); i++) {
        //TODO write encoding here in 'encoded' form into the variable 'qcMapped'
        //i.e. which operations do you need to perform to get an "all |0>" state?

        //if you do not encode each qubit for itself, feel free to remove the loop :)
    }
}

void QxCustomEcc::measureAndCorrect() {
    for (std::size_t i = 0; i < qcOriginal->getNqubits(); i++) {
        //TODO write correcting here in 'encoded' form into the variable 'qcMapped'
        //i.e. which operations do you need to perform to get correct a state in case of errors?

        //if you do not encode each qubit for itself, feel free to remove the loop :)
    }
}

void QxCustomEcc::writeDecoding() {
    for (std::size_t i = 0; i < qcOriginal->getNqubits(); i++) {
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
