/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

//Reference to this ecc in https://arxiv.org/pdf/1608.05053.pdf

class Q9Surface: public Ecc {
public:
    Q9Surface(std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFq):
        Ecc({ID::Q9Surface, 9, 8, "Q9Surface"}, std::move(qc), measureFq) {}

protected:
    void initMappedCircuit() override;

    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;

    // Set parameter for verifying the ecc
    [[maybe_unused]] const size_t insertErrorAfterNGates = 55;
};
