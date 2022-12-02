/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#ifndef QFR_QxCustomEcc_HPP
#define QFR_QxCustomEcc_HPP

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class QxCustomEcc: public Ecc {
public:
    //TODO parameters q and c:
    /*
 * q = #qubits
 * c = #classical bits
 * Assume your ECC needs p physical qubits to encode 1 logical qubit, a ancilla qubits and m measurements.
 * >>then q = p+a and c=m.
 */
    QxCustomEcc(qc::QuantumComputation& qc, int measureFq):
        Ecc({ID::QxCustom, /*q*/ -1, /*c*/ -1, "QxCustom"}, qc, measureFq) {}

protected:
    void writeEncoding() override;

    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;
};

#endif //QFR_QxCustomEcc_HPP
