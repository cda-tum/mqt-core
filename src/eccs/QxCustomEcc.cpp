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
QxCustomEcc::QxCustomEcc(qc::QuantumComputation& qc): Ecc({EccID::QxCustom, /*q*/-1, /*c*/-1, QxCustomEcc::getEccName()}, qc) {}

void QxCustomEcc::writeEccEncoding() {
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

void QxCustomEcc::writeEccDecoding() {
    const int nQubits = qc.getNqubits();
    for(int i=0;i<nQubits;i++) {
        //TODO write correcting here in 'encoded' form into the variable 'qcMapped'
        //i.e. which operations do you need to perform to get correct a state in case of errors?

        //if you do not encode each qubit for itself, feel free to remove the loop :)
    }
}

void QxCustomEcc::mapGate(std::unique_ptr<qc::Operation> &gate) {

    //TODO make sure the parameter gate is written in 'encoded' form into the variable 'qcMapped'

    //for error cases, use the following lines
    statistics.nOutputGates = -1;
    statistics.nOutputQubits = -1;
    throw qc::QFRException("Gate not possible to encode in error code!");
}
