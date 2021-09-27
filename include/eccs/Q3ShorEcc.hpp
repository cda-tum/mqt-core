/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"
#include "Ecc.hpp"

#ifndef QFR_Q3ShorEcc_HPP
#define QFR_Q3ShorEcc_HPP

class Q3ShorEcc: public Ecc {
public:
    Q3ShorEcc(qc::QuantumComputation& qc);

    static const std::string getEccName() {
        return "Q3Shor";
    }

protected:
    void writeEccEncoding() override;

    void measureAndCorrect() override;

	void writeEccDecoding() override;

	void mapGate(std::unique_ptr<qc::Operation> &gate) override;
};

#endif //QFR_Q3ShorEcc_HPP
