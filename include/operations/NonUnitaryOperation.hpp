/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef INTERMEDIATEREPRESENTATION_NONUNITARYOPERATION_H
#define INTERMEDIATEREPRESENTATION_NONUNITARYOPERATION_H

#include "Operation.hpp"

namespace qc {

	class NonUnitaryOperation : public Operation {

	public:
		// Measurement constructor
		NonUnitaryOperation(unsigned short nq, const std::vector<unsigned short>& qubitRegister, const std::vector<unsigned short>& classicalRegister);

		// Snapshot constructor
		NonUnitaryOperation(unsigned short nq, const std::vector<unsigned short>& qubitRegister, int n);

		// ShowProbabilities constructor
		explicit NonUnitaryOperation(const unsigned short nq) {
			nqubits = nq;
			type = ShowProbabilities;
		}

		// General constructor
		NonUnitaryOperation(unsigned short nq, const std::vector<unsigned short>& qubitRegister, OpType op = Reset);

		dd::Edge getDD(std::unique_ptr<dd::Package>&, std::array<short, MAX_QUBITS>&) const override {
			std::cerr << "DD for non-unitary operation not available!" << std::endl;
			exit(1);
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const override {
			return getDD(dd, line);
		}

		dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>&) const override {
			return getDD(dd, line);
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>&) const override {
			return getInverseDD(dd, line);
		}

		bool isUnitary() const override {
			return false;
		}

		bool isNonUnitaryOperation() const override {
			return true;
		}

		bool actsOn(unsigned short i) override {
			if (type != Measure) {
				for (const auto t:targets) {
					if (t == i)
						return true;
				}
			} else {
				for (const auto c:controls) {
					if (c.qubit == i)
						return true;
				}
			}
			return false;
		}

		std::ostream& print(std::ostream& os) const override;
		
		void dumpOpenQASM(std::ofstream& of, const regnames_t& qreg, const regnames_t& creg) const override;
		void dumpQiskit(std::ofstream& of, const regnames_t& qreg, const regnames_t& creg, const char *anc_reg_name) const override;
		void dumpReal(std::ofstream& of) const override {
			UNUSED(of)// these ops do not exist in .real
		};

		std::ostream& print(std::ostream& os, const std::map<unsigned short, unsigned short>& permutation) const override;
	};
}
#endif //INTERMEDIATEREPRESENTATION_NONUNITARYOPERATION_H
