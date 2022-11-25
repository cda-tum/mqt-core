/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_Q5LaflammeEcc_HPP
#define QFR_Q5LaflammeEcc_HPP

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class Q5LaflammeEcc: public Ecc {
public:
    Q5LaflammeEcc(qc::QuantumComputation& qc, int measureFq);

    static std::string getName() {
        return "Q5Laflamme";
    }

protected:
    void initMappedCircuit() override;

    void writeEncoding() override;

    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;

private:
    void writeClassicalControlled(unsigned int value, int target, qc::OpType optype, dd::Qubit clStart, dd::QubitCount clCount);
    void writeClassicalControlledCorrect(unsigned int value, int target, qc::OpType optype);
};

#endif //QFR_Q5LaflammeEcc_HPP
