/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"
#include "EccStatistics.hpp"

#ifndef QFR_Ecc_HPP
#define QFR_Ecc_HPP

class Ecc {
public:

    enum ID {
		Id, Q3Shor, Q9Shor, Q7Steane, QxCustom
	};
    struct Info {
        EccID enumID;
        int nRedundantQubits;
        int nClassicalBits;
        std::string name;
    };

    const struct EccInfo ecc;

	Ecc(struct EccInfo ecc_type, qc::QuantumComputation& qc);
	virtual ~Ecc() = default;

	qc::QuantumComputation& applyEcc();

    virtual std::ostream& printResult(std::ostream& out);

    virtual void dumpResult(const std::string& outputFilename);

	virtual void dumpResult(const std::string& outputFilename, qc::Format format) {
		size_t slash = outputFilename.find_last_of('/');
		size_t dot = outputFilename.find_last_of('.');
		statistics.outputName = outputFilename.substr(slash+1, dot-slash-1);
		qcMapped.dump(outputFilename, format);
	}

	virtual void dumpResult(std::ostream& os, qc::Format format) {
		qcMapped.dump(os, format);
	}


protected:

    qc::QuantumComputation& qc;
	qc::QuantumComputation qcMapped;
	EccStatistics statistics{};

	virtual void writeEccEncoding()=0;

	virtual void measureAndCorrect()=0;

	virtual void writeEccDecoding()=0;

	virtual void mapGate(std::unique_ptr<qc::Operation> &gate)=0;

    void writeToffoli(unsigned short c1, unsigned short c2, unsigned short target);
    void writeCnot(unsigned short control, unsigned short target);

    dd::Control createControl(unsigned short qubit, bool pos);

};


#endif //QFR_Ecc_HPP
