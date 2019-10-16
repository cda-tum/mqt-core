//
// Created by Lukas Burgholzer on 22.10.19.
//

#ifndef INTERMEDIATEREPRESENTATION_NONUNITARYOPERATION_H
#define INTERMEDIATEREPRESENTATION_NONUNITARYOPERATION_H

#include <cstring>
#include <memory>
#include <iostream>

#include "Operation.h"

namespace qc {

	enum Op : short {
		Measure, Reset, Snapshot, ShowProbabilities
	};

	class NonUnitaryOperation : public Operation {
	protected:
		Op op = ShowProbabilities;
	public:
		// Measurement constructor
		NonUnitaryOperation(const unsigned short nq, const std::vector<unsigned short>& qubitRegister, const std::vector<unsigned short>& classicalRegister) : op(Measure) {
			nqubits = nq;
			strcpy(name, "Measure");
			line.fill(-1);
			// i-th qubit to be measured shall be measured into i-th classical register
			for (unsigned long i = 0; i < qubitRegister.size(); ++i) {
				line[qubitRegister[i]] = classicalRegister[i];
			}

			parameter[0] = qubitRegister.size();
		}

		// Reset constructor
		NonUnitaryOperation(const unsigned short nq, const std::vector<unsigned short>& qubitRegister) : op(Reset) {
			nqubits = nq;
			strcpy(name, "Reset");
			line.fill(-1);

			// mark qubits
			for (unsigned short q: qubitRegister) {
				line[q] = 2;
			}
			parameter[0] = qubitRegister.size();
		}

		// Snapshot constructor
		NonUnitaryOperation(const unsigned short nq, const std::vector<unsigned short>& qubitRegister, int n) : op(Snapshot) {
			nqubits = nq;
			strcpy(name, "Snapshot");
			line.fill(-1);

			// mark qubits
			for (short q: qubitRegister) {
				line[q] = 2;
			}
			parameter[0] = qubitRegister.size();
			parameter[1] = n;
		}

		// ShowProbabilities constructor
		explicit NonUnitaryOperation(const unsigned short nq) : op(ShowProbabilities) {
			nqubits = nq;
		}

		dd::Edge getDD(std::unique_ptr<dd::Package>&) override {
			std::cerr << "DD for non-unitary operation not available!" << std::endl;
			exit(1);
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>&) override {
			std::cerr << "DD for non-unitary operation not available!" << std::endl;
			exit(1);
		}

		bool isUnitary() const override { return false; }


		std::ostream& print(std::ostream& os) const override {
			switch (op) {
				case Measure: os << "Meas ";
					for (int i = 0; i < nqubits; ++i) {
						if (line[i] >= 0) {
							os << "\033[34m" << line[i] << " " << "\033[0m";
						} else {
							os << "| ";
						}
					}
					break;
				case Reset: os << "Rst  ";
					for (int i = 0; i < nqubits; ++i) {
						if (line[i] == 2) {
							os << "\033[31m" << "r " << "\033[0m";
						} else {
							os << "| ";
						}
					}
					break;
				case Snapshot: os << "Snap ";
					for (int i = 0; i < nqubits; ++i) {
						if (line[i] == 2) {
							os << "\033[33m" << "s " << "\033[0m";
						} else {
							os << "| ";
						}
					}
					os << "\tp: " << parameter[0] << " " << parameter[1];
					break;
				case ShowProbabilities: os << "Show probabilities";
					break;
			}
			return os;
		}
	};
}
#endif //INTERMEDIATEREPRESENTATION_NONUNITARYOPERATION_H
