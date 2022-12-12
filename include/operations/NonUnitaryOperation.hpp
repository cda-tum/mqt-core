/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "Operation.hpp"

namespace qc {

    class NonUnitaryOperation final: public Operation {
    protected:
        std::vector<dd::Qubit>   qubits{};   // vector for the qubits to measure (necessary since std::set does not preserve the order of inserted elements)
        std::vector<std::size_t> classics{}; // vector for the classical bits to measure into

        std::ostream& printNonUnitary(std::ostream& os, const std::vector<dd::Qubit>& q, const std::vector<std::size_t>& c = {}, const Permutation& permutation = {}) const;
        void printMeasurement(std::ostream& os, const std::vector<dd::Qubit>& q, const std::vector<std::size_t>& c, const Permutation& permutation) const;
        void printResetBarrierOrSnapshot(std::ostream& os, const std::vector<dd::Qubit>& q, const Permutation& permutation) const;

    public:
        // Measurement constructor
        NonUnitaryOperation(dd::QubitCount nq, std::vector<dd::Qubit> qubitRegister, std::vector<std::size_t> classicalRegister);
        NonUnitaryOperation(dd::QubitCount nq, dd::Qubit qubit, std::size_t cbit);

        // Snapshot constructor
        NonUnitaryOperation(dd::QubitCount nq, const std::vector<dd::Qubit>& qubitRegister, std::size_t n);

        // ShowProbabilities constructor
        explicit NonUnitaryOperation(const dd::QubitCount nq) {
            nqubits = nq;
            type    = ShowProbabilities;
        }

        // General constructor
        NonUnitaryOperation(dd::QubitCount nq, const std::vector<dd::Qubit>& qubitRegister, OpType op = Reset);

        [[nodiscard]] std::unique_ptr<Operation> clone() const override {
            if (getType() == qc::Measure) {
                return std::make_unique<NonUnitaryOperation>(getNqubits(), getTargets(), getClassics());
            } else if (getType() == qc::Snapshot) {
                return std::make_unique<NonUnitaryOperation>(getNqubits(), getTargets(), getParameter().at(0));
            } else if (getType() == qc::ShowProbabilities) {
                return std::make_unique<NonUnitaryOperation>(getNqubits());
            } else {
                return std::make_unique<NonUnitaryOperation>(getNqubits(), getTargets(), getType());
            }
        }

        [[nodiscard]] bool isUnitary() const override {
            return false;
        }

        [[nodiscard]] bool isNonUnitaryOperation() const override {
            return true;
        }

        [[nodiscard]] const Targets& getTargets() const override {
            if (type == Measure)
                return qubits;
            else
                return targets;
        }
        Targets& getTargets() override {
            if (type == Measure)
                return qubits;
            else
                return targets;
        }
        [[nodiscard]] size_t getNtargets() const override {
            return getTargets().size();
        }

        [[nodiscard]] const std::vector<std::size_t>& getClassics() const {
            return classics;
        }
        std::vector<std::size_t>& getClassics() {
            return classics;
        }
        [[nodiscard]] size_t getNclassics() const {
            return classics.size();
        }

        [[nodiscard]] bool actsOn(dd::Qubit i) const override;

        void addDepthContribution(std::vector<std::size_t>& depths) const override;

        [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1, const Permutation& perm2) const override;
        [[nodiscard]] bool equals(const Operation& operation) const override {
            return equals(operation, {}, {});
        }

        std::ostream& print(std::ostream& os) const override {
            if (type == Measure) {
                return printNonUnitary(os, qubits, classics);
            } else {
                return printNonUnitary(os, targets);
            }
        }
        std::ostream& print(std::ostream& os, const Permutation& permutation) const override {
            if (type == Measure) {
                return printNonUnitary(os, qubits, classics, permutation);
            } else {
                return printNonUnitary(os, targets, {}, permutation);
            }
        }

        void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg) const override;

        [[nodiscard]] std::set<dd::Qubit> getUsedQubits() const override {
            const auto& targets = getTargets();
            return std::set<dd::Qubit>{targets.begin(), targets.end()};
        }
    };
} // namespace qc
