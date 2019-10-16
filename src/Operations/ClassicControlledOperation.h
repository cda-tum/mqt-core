//
// Created by Lukas Burgholzer on 22.10.19.
//

#ifndef INTERMEDIATEREPRESENTATION_CLASSICCONTROLLEDOPERATION_H
#define INTERMEDIATEREPRESENTATION_CLASSICCONTROLLEDOPERATION_H

#include "Operation.h"

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

		dd::Edge getDD(std::unique_ptr<dd::Package>& dd) override {
			return op->getDD(dd);
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd) override {
			return op->getInverseDD(dd);
		}

		bool isUnitary() const override {
			return false;
		}
	};
}
#endif //INTERMEDIATEREPRESENTATION_CLASSICCONTROLLEDOPERATION_H
