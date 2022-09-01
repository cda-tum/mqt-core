/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_Q9ShorEcc_HPP
#define QFR_Q9ShorEcc_HPP

#include "QuantumComputation.hpp"
#include "Ecc.hpp"

class Q9ShorEcc: public Ecc {
public:
    Q9ShorEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly);

    static const std::string getName() {
        return "Q9Shor";
    }

protected:
    void initMappedCircuit() override;

    void writeEncoding() override;

    void measureAndCorrect() override;

	void writeDecoding() override;

    void mapGate(const std::unique_ptr<qc::Operation>& gate, qc::QuantumComputation& qc) override;

};

#endif //QFR_Q9ShorEcc_HPP
