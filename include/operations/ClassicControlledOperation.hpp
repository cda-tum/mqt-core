/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "Operation.hpp"

namespace qc {

    class ClassicControlledOperation final: public Operation {
    private:
        std::unique_ptr<Operation> op;
        ClassicalRegister          controlRegister{};
        std::uint64_t              expectedValue = 1U;

    public:
        // Applies operation `_op` if the creg starting at index `control` has the expected value
        ClassicControlledOperation(std::unique_ptr<qc::Operation>& op, const ClassicalRegister& controlRegister, std::uint64_t expectedValue = 1U):
            op(std::move(op)), controlRegister(controlRegister), expectedValue(expectedValue) {
            nqubits      = this->op->getNqubits();
            name         = "c_" + this->op->getName();
            parameter[0] = static_cast<fp>(controlRegister.first);
            parameter[1] = static_cast<fp>(controlRegister.second);
            parameter[2] = static_cast<fp>(expectedValue);
            type         = ClassicControlled;
        }

        [[nodiscard]] std::unique_ptr<Operation> clone() const override {
            auto opCloned = op->clone();
            return std::make_unique<ClassicControlledOperation>(opCloned, controlRegister, expectedValue);
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

        void setNqubits(std::size_t nq) override {
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

        [[nodiscard]] const Controls& getControls() const override {
            return op->getControls();
        }

        Controls& getControls() override {
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

        [[nodiscard]] bool actsOn(Qubit i) const override {
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
            }
            return false;
        }
        [[nodiscard]] bool equals(const Operation& operation) const override {
            return equals(operation, {}, {});
        }

        void dumpOpenQASM([[maybe_unused]] std::ostream& of, [[maybe_unused]] const RegisterNames& qreg, [[maybe_unused]] const RegisterNames& creg) const override {
            throw QFRException("Dumping of classically controlled gates currently not supported for qasm");
        }
    };
} // namespace qc
