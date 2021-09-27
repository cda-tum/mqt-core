/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"
#include "Ecc.hpp"


#ifndef QFR_IdEcc_HPP
#define QFR_IdEcc_HPP

class IdEcc: public Ecc {
public:
    IdEcc(qc::QuantumComputation& qc);

    static const std::string getEccName() {
        return "Id";
    }

protected:
    void writeEccEncoding() override;

    void measureAndCorrect() override;

	void writeEccDecoding() override;

	void mapGate(std::unique_ptr<qc::Operation> &gate) override;
};

#endif //QFR_IdEcc_HPP


