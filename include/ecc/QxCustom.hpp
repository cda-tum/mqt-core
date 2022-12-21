/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class QxCustom: public Ecc {
public:
    //TODO parameters q and c:
    /*
 * q = #qubits
 * c = #classical bits
 * Assume your ECC needs p physical qubits to encode 1 logical qubit, a ancilla qubits and m measurements.
 * >>then q = p+a and c=m.
 */
    QxCustom(std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFq):
        Ecc({ID::QxCustom, /*q*/ 0, /*c*/ 0, "QxCustom"}, std::move(qc), measureFq) {}

protected:
    void writeEncoding() override;

    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;
};
