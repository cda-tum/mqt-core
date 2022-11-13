/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_IdEcc_HPP
#define QFR_IdEcc_HPP

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class IdEcc: public Ecc {
public:
    IdEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly);

    static const std::string getName() {
        return "Id";
    }

protected:
    void writeEncoding() override;

    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const std::unique_ptr<qc::Operation>& gate, qc::QuantumComputation& qc) override;
};

#endif //QFR_IdEcc_HPP
