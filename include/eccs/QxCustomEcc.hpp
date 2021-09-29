/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"
#include "Ecc.hpp"

#ifndef QFR_QxCustomEcc_HPP
#define QFR_QxCustomEcc_HPP

class QxCustomEcc: public Ecc {
public:
    QxCustomEcc(qc::QuantumComputation& qc);

    static const std::string getEccName() {
        return "QxCustom";
    }

protected:
    void writeEccEncoding() override;

    void measureAndCorrect() override;

	void writeEccDecoding() override;

	void mapGate(std::unique_ptr<qc::Operation> &gate) override;
};

#endif //QFR_QxCustomEcc_HPP
