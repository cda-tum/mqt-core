/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_NONUNITARYOPERATION_H
#define QFR_NONUNITARYOPERATION_H

#include "Operation.hpp"

namespace qc {

    class NonUnitaryOperation final: public Operation {
    protected:
        std::vector<dd::Qubit>   qubits{};   // vector for the qubits to measure (necessary since std::set does not preserve the order of inserted elements)
        std::vector<std::size_t> classics{}; // vector for the classical bits to measure into

        std::ostream& printNonUnitary(std::ostream& os, const std::vector<dd::Qubit>& q, const std::vector<std::size_t>& c = {}) const;

        MatrixDD getDD(std::unique_ptr<dd::Package>& dd, const dd::Controls& controls, const Targets& targets) const override;
        MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd, const dd::Controls& controls, const Targets& targets) const override;

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
                return printNonUnitary(os, permutation.apply(qubits), classics);
            } else {
                return printNonUnitary(os, permutation.apply(targets));
            }
        }

        MatrixDD getDD(std::unique_ptr<dd::Package>& dd) const override {
            return Operation::getDD(dd);
        }
        MatrixDD getDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const override {
            return Operation::getDD(dd, permutation);
        }
        MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd) const override {
            return Operation::getInverseDD(dd);
        }
        MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const override {
            return Operation::getInverseDD(dd, permutation);
        }

        void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg) const override;
        void dumpQiskit(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg, const char* anc_reg_name) const override;
        void dumpTensor(std::ostream& of, std::vector<std::size_t>& inds, std::size_t& gateIdx, std::unique_ptr<dd::Package>& dd) override;
    };
} // namespace qc
#endif //QFR_NONUNITARYOPERATION_H
