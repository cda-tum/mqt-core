/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"
#include "Ecc.hpp"


#ifndef QFR_Q9ShorEcc_HPP
#define QFR_Q9ShorEcc_HPP

class Q9ShorEcc: public Ecc {
public:
    Q9ShorEcc(qc::QuantumComputation& qc);

    static const std::string getEccName() {
        return "Q9Shor";
    }

protected:
    void writeEccEncoding();

	void writeEccDecoding();

	void mapGate(std::unique_ptr<qc::Operation> &gate) override;
};

#endif //QFR_Q9ShorEcc_HPP
