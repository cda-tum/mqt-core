/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class Q3Shor: public Ecc {
public:
    Q3Shor(std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFq):
        Ecc(
                {ID::Q3Shor, 3, 2, "Q3Shor"}, std::move(qc), measureFq) {}

protected:
    void initMappedCircuit() override;

    void writeEncoding() override;

    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;
};
