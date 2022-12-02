/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class IdEcc: public Ecc {
public:
    IdEcc(qc::QuantumComputation& qc, int measureFq):
        Ecc({ID::Id, 1, 0, "Id"}, qc, measureFq) {}

protected:
    void writeEncoding() override{};

    void measureAndCorrect() override{};

    void writeDecoding() override{};

    void mapGate(const qc::Operation& gate) override;
};
