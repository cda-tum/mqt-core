/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#ifndef QFR_Q5LaflammeEcc_HPP
#define QFR_Q5LaflammeEcc_HPP

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class Q5LaflammeEcc: public Ecc {
public:
    Q5LaflammeEcc(qc::QuantumComputation& qc, int measureFq):
        Ecc({ID::Q5Laflamme, 5, 4, Q5LaflammeEcc::getName()}, qc, measureFq) {}

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
