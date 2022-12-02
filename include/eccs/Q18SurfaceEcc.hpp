/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class Q18SurfaceEcc: public Ecc {
public:
    Q18SurfaceEcc(qc::QuantumComputation& qc, int measureFq):
        Ecc({ID::Q18Surface, 36, 0, "Q18Surface"}, qc, measureFq) {}

protected:
    void initMappedCircuit() override;

    void writeEncoding() override;

    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;
};
