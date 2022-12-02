/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#ifndef QFR_Q3ShorEcc_HPP
#define QFR_Q3ShorEcc_HPP

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class Q3ShorEcc: public Ecc {
public:
    Q3ShorEcc(qc::QuantumComputation& qc, int measureFq):
        Ecc(
                {ID::Q3Shor, 3, 2, "Q3Shor"}, qc, measureFq) {}

protected:
    void initMappedCircuit() override;

    void writeEncoding() override;

    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;
};

#endif //QFR_Q3ShorEcc_HPP
