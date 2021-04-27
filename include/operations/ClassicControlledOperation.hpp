/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_CLASSICCONTROLLEDOPERATION_H
#define QFR_CLASSICCONTROLLEDOPERATION_H

#include "Operation.hpp"

namespace qc {

    class ClassicControlledOperation final: public Operation {
    protected:
        std::unique_ptr<Operation>           op;
        std::pair<dd::Qubit, dd::QubitCount> controlRegister{};
        unsigned int                         expectedValue = 1U;

        MatrixDD getDD([[maybe_unused]] std::unique_ptr<dd::Package>& dd, [[maybe_unused]] const dd::Controls& controls, [[maybe_unused]] const Targets& targets) const override {
            throw QFRException("[ClassicControlledOperation] protected getDD called which should not happen.");
        }
        MatrixDD getInverseDD([[maybe_unused]] std::unique_ptr<dd::Package>& dd, [[maybe_unused]] const dd::Controls& controls, [[maybe_unused]] const Targets& targets) const override {
            throw QFRException("[ClassicControlledOperation] protected getInverseDD called which should not happen.");
        }

    public:
        // Applies operation `_op` if the creg starting at index `control` has the expected value
        ClassicControlledOperation(std::unique_ptr<Operation>& _op, const std::pair<dd::Qubit, dd::QubitCount>& controlRegister, unsigned int expectedValue = 1U):
            op(std::move(_op)), controlRegister(controlRegister), expectedValue(expectedValue) {
            nqubits = op->getNqubits();
            name[0] = 'c';
            name[1] = '_';
            std::strcpy(name + 2, op->getName());
            parameter[0] = controlRegister.first;
            parameter[1] = controlRegister.second;
            parameter[2] = expectedValue;
            type         = ClassicControlled;
        }

        MatrixDD getDD(std::unique_ptr<dd::Package>& dd) const override {
            return op->getDD(dd);
        }

        MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd) const override {
            return op->getInverseDD(dd);
        }

        MatrixDD getDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const override {
            return op->getDD(dd, permutation);
        }

        MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const override {
            return op->getInverseDD(dd, permutation);
        }

        [[nodiscard]] auto getControlRegister() const {
            return controlRegister;
        }

        [[nodiscard]] auto getExpectedValue() const {
            return expectedValue;
        }

        [[nodiscard]] auto getOperation() const {
            return op.get();
        }

        [[nodiscard]] bool isUnitary() const override {
            return false;
        }

        [[nodiscard]] bool isClassicControlledOperation() const override {
            return true;
        }

        [[nodiscard]] bool actsOn(dd::Qubit i) const override {
            return op->actsOn(i);
        }

        void dumpOpenQASM([[maybe_unused]] std::ostream& of, [[maybe_unused]] const RegisterNames& qreg, [[maybe_unused]] const RegisterNames& creg) const override {
            throw QFRException("Dumping of classically controlled gates currently not supported for qasm");
        }

        void dumpQiskit([[maybe_unused]] std::ostream& of, [[maybe_unused]] const RegisterNames& qreg, [[maybe_unused]] const RegisterNames& creg, [[maybe_unused]] const char* anc_reg_name) const override {
            throw QFRException("Dumping of classically controlled gates currently not supported for qiskit");
        }
    };
} // namespace qc
#endif //QFR_CLASSICCONTROLLEDOPERATION_H
