//
// Created by Lukas Burgholzer on 22.10.19.
//

#ifndef INTERMEDIATEREPRESENTATION_NONUNITARYOPERATION_H
#define INTERMEDIATEREPRESENTATION_NONUNITARYOPERATION_H

#include <cstring>
#include <memory>
#include <iostream>

#include "Operation.hpp"

namespace qc {

	enum Op : short {
		Measure, Reset, Snapshot, ShowProbabilities
	};

	class NonUnitaryOperation : public Operation {
	protected:
		Op op = ShowProbabilities;
	public:
		// Measurement constructor
		NonUnitaryOperation(unsigned short nq, const std::vector<unsigned short>& qubitRegister, const std::vector<unsigned short>& classicalRegister);

		// Reset constructor
		NonUnitaryOperation(unsigned short nq, const std::vector<unsigned short>& qubitRegister);

		// Snapshot constructor
		NonUnitaryOperation(unsigned short nq, const std::vector<unsigned short>& qubitRegister, int n);

		// ShowProbabilities constructor
		explicit NonUnitaryOperation(const unsigned short nq) : op(ShowProbabilities) {
			nqubits = nq;
		}

		dd::Edge getDD(std::unique_ptr<dd::Package>&, std::array<short, MAX_QUBITS>& line) override {
			(void)line;
			std::cerr << "DD for non-unitary operation not available!" << std::endl;
			exit(1);
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>&, std::array<short, MAX_QUBITS>& line) override {
			(void)line;
			std::cerr << "DD for non-unitary operation not available!" << std::endl;
			exit(1);
		}

		bool isUnitary() const override { 
			return false;
		}
		
		std::ostream& print(std::ostream& os) const override;
		
		void dumpOpenQASM(std::ofstream& of, const std::vector<std::string>& qreg, const std::vector<std::string>& creg) const override;
	};
}
#endif //INTERMEDIATEREPRESENTATION_NONUNITARYOPERATION_H
