/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "Operation.hpp"

namespace qc {

    class ClassicControlledOperation final: public Operation {
    protected:
        std::unique_ptr<Operation>           op;
        std::pair<dd::Qubit, dd::QubitCount> controlRegister{};
        unsigned int                         expectedValue = 1U;

    public:
        // Applies operation `_op` if the creg starting at index `control` has the expected value
        ClassicControlledOperation(std::unique_ptr<qc::Operation>& _op, const std::pair<dd::Qubit, dd::QubitCount>& controlRegister, unsigned int expectedValue = 1U):
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

        [[nodiscard]] std::unique_ptr<Operation> clone() const override {
            auto op_cloned = op->clone();
            return std::make_unique<ClassicControlledOperation>(op_cloned, controlRegister, expectedValue);
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

        void setNqubits(dd::QubitCount nq) override {
            nqubits = nq;
            op->setNqubits(nq);
        }

        [[nodiscard]] const Targets& getTargets() const override {
            return op->getTargets();
        }

        Targets& getTargets() override {
            return op->getTargets();
        }

        [[nodiscard]] std::size_t getNtargets() const override {
            return op->getNtargets();
        }

        [[nodiscard]] const dd::Controls& getControls() const override {
            return op->getControls();
        }

        dd::Controls& getControls() override {
            return op->getControls();
        }

        [[nodiscard]] std::size_t getNcontrols() const override {
            return controls.size();
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

        [[nodiscard]] bool equals(const Operation& operation, const Permutation& perm1, const Permutation& perm2) const override {
            if (const auto* classic = dynamic_cast<const ClassicControlledOperation*>(&operation)) {
                if (controlRegister != classic->controlRegister) {
                    return false;
                }

                if (expectedValue != classic->expectedValue) {
                    return false;
                }

                return op->equals(*classic->op, perm1, perm2);

            } else {
                return false;
            }
            return Operation::equals(operation, perm1, perm2);
        }
        [[nodiscard]] bool equals(const Operation& operation) const override {
            return equals(operation, {}, {});
        }

        void dumpOpenQASM([[maybe_unused]] std::ostream& of, [[maybe_unused]] const RegisterNames& qreg, [[maybe_unused]] const RegisterNames& creg) const override {
            of << "if(";
            of << creg[controlRegister.first].first;
            of << " == " << expectedValue << ") ";
            op->dumpOpenQASM(of, qreg, creg);
        }

        void dumpQiskit([[maybe_unused]] std::ostream& of, [[maybe_unused]] const RegisterNames& qreg, [[maybe_unused]] const RegisterNames& creg, [[maybe_unused]] const char* anc_reg_name) const override {
            throw QFRException("Dumping of classically controlled gates currently not supported for qiskit");
        }
    };
} // namespace qc
