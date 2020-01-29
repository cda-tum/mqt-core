/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef INTERMEDIATEREPRESENTATION_CLASSICCONTROLLEDOPERATION_H
#define INTERMEDIATEREPRESENTATION_CLASSICCONTROLLEDOPERATION_H

#include "Operation.hpp"

namespace qc {

	class ClassicControlledOperation : public Operation {
	protected:
		std::unique_ptr<Operation> op;
		short control;
	public:

		ClassicControlledOperation(std::unique_ptr<Operation>& op, short control) : op(std::move(op)), control(control) {
			nqubits = op->getNqubits();
			name[0] = 'c';
			name[1] = '_';
			std::strcpy(name + 2, op->getName());
			parameter[0] = control;
		}

		dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const override {
			return op->getDD(dd, line);
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const override {
			return op->getInverseDD(dd, line);
		}

		dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const override {
			return op->getDD(dd, line, permutation);
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const override {
			return op->getInverseDD(dd, line, permutation);
		}

		bool isUnitary() const override {
			return false;
		}
	};
}
#endif //INTERMEDIATEREPRESENTATION_CLASSICCONTROLLEDOPERATION_H
