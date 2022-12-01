/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_Q9SurfaceEcc_HPP
#define QFR_Q9SurfaceEcc_HPP

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

//Reference to this ecc in https://arxiv.org/pdf/1608.05053.pdf

class Q9SurfaceEcc: public Ecc {
public:
    Q9SurfaceEcc(qc::QuantumComputation& qc, int measureFq):
        Ecc({ID::Q9Surface, 9, 8, "Q9Surface"}, qc, measureFq) {}

protected:
    void initMappedCircuit() override;

    void writeEncoding() override;

    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;
};

#endif //QFR_Q9SurfaceEcc_HPP
