/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class Q5Laflamme: public Ecc {
public:
    Q5Laflamme(std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFq):
        Ecc({ID::Q5Laflamme, 5, 4, "Q5Laflamme"}, std::move(qc), measureFq) {}

protected:
    void initMappedCircuit() override;

    void writeEncoding() override;

    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;

private:
    void writeClassicalControlled(unsigned int value, int target, qc::OpType opType, dd::Qubit clStart, dd::QubitCount clCount);
    void writeClassicalControlledCorrect(unsigned int value, int target, qc::OpType opType);
};
