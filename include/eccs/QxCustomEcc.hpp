/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_QxCustomEcc_HPP
#define QFR_QxCustomEcc_HPP

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class QxCustomEcc: public Ecc {
public:
    QxCustomEcc(qc::QuantumComputation& qc, int measureFq);

    static std::string getName() {
        return "QxCustom";
    }

protected:
    void writeEncoding() override;

    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;
};

#endif //QFR_QxCustomEcc_HPP
