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
		std::pair<unsigned short, unsigned short> controlRegister{};
		unsigned int expectedValue = 1u;
	public:

		// Applies operation `_op` if the creg starting at index `control` has the expected value
		ClassicControlledOperation(std::unique_ptr<Operation>& _op, std::pair<unsigned short, unsigned short>& controlRegister, unsigned int expectedValue = 1u) : op(std::move(_op)), controlRegister(controlRegister), expectedValue(expectedValue) {
			nqubits = op->getNqubits();
			name[0] = 'c';
			name[1] = '_';
			std::strcpy(name + 2, op->getName());
			parameter[0] = controlRegister.first;
			parameter[1] = controlRegister.second;
			parameter[2] = expectedValue;
			type = ClassicControlled;
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

		auto getControlRegister() const {
			return controlRegister;
		}

		auto getExpectedValue() const {
			return expectedValue;
		}

		auto getOperation() const {
			return op.get();
		}

		bool isUnitary() const override {
			return false;
		}

		bool isClassicControlledOperation() const override {
			return true;
		}

		bool actsOn(unsigned short i) override {
			return op->actsOn(i);
		}

		void setLine(std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const override {
			op->setLine(line, permutation);
		}

		void resetLine(std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const override {
			op->resetLine(line, permutation);
		}

		void dumpOpenQASM(std::ostream& of, const regnames_t& qreg, const regnames_t& creg) const override {
			UNUSED(of)
			UNUSED(qreg)
			UNUSED(creg)
			throw QFRException("Dumping of classically controlled gates currently not supported for qasm");
		}

		void dumpReal(std::ostream& of) const override {
			UNUSED(of)
			throw QFRException("Dumping of classically controlled gates not possible in real format");
		}

		void dumpQiskit(std::ostream& of, const regnames_t& qreg, const regnames_t& creg, const char *anc_reg_name) const override {
			UNUSED(of)
			UNUSED(qreg)
			UNUSED(creg)
			UNUSED(anc_reg_name)
			throw QFRException("Dumping of classically controlled gates currently not supported for qiskit");
		}
	};
}
#endif //INTERMEDIATEREPRESENTATION_CLASSICCONTROLLEDOPERATION_H
