/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_Q7SteaneEcc_HPP
#define QFR_Q7SteaneEcc_HPP

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class Q7SteaneEcc: public Ecc {
public:
    Q7SteaneEcc(qc::QuantumComputation& qc, int measureFq);

    static std::string getName() {
        return "Q7Steane";
    }

protected:
    void initMappedCircuit() override;

    void writeEncoding() override;

    void measureAndCorrect() override;
    void measureAndCorrectSingle(bool xSyndrome);

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;
};

#endif //QFR_Q7SteaneEcc_HPP
