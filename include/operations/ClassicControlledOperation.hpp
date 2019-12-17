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
			strcpy(name + 2, op->getName());
			parameter[0] = control;
		}

		dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) override {
			return op->getDD(dd, line);
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) override {
			return op->getInverseDD(dd, line);
		}

		bool isUnitary() const override {
			return false;
		}
	};
}
#endif //INTERMEDIATEREPRESENTATION_CLASSICCONTROLLEDOPERATION_H
