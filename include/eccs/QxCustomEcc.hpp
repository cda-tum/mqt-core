/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_QxCustomEcc_HPP
#define QFR_QxCustomEcc_HPP

#include "QuantumComputation.hpp"
#include "Ecc.hpp"

class QxCustomEcc: public Ecc {
public:
    QxCustomEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly);

    static const std::string getName() {
        return "QxCustom";
    }

protected:
    void writeEncoding() override;

    void measureAndCorrect() override;

	void writeDecoding() override;

    void mapGate(const std::unique_ptr<qc::Operation>& gate, qc::QuantumComputation& qc) override;
};

#endif //QFR_QxCustomEcc_HPP
