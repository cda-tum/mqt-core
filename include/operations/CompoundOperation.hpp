/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_COMPOUNDOPERATION_H
#define QFR_COMPOUNDOPERATION_H

#include "Operation.hpp"

namespace qc {

    class CompoundOperation final: public Operation {
    protected:
        std::vector<std::unique_ptr<Operation>> ops{};

        MatrixDD getDD([[maybe_unused]] std::unique_ptr<dd::Package>& dd, [[maybe_unused]] const dd::Controls& controls, [[maybe_unused]] const Targets& targets) const override {
            throw QFRException("[CompoundOperation] protected getDD called which should not happen.");
        }
        MatrixDD getInverseDD([[maybe_unused]] std::unique_ptr<dd::Package>& dd, [[maybe_unused]] const dd::Controls& controls, [[maybe_unused]] const Targets& targets) const override {
            throw QFRException("[CompoundOperation] protected getInverseDD called which should not happen.");
        }

    public:
        explicit CompoundOperation(dd::QubitCount nq) {
            std::strcpy(name, "Compound operation:");
            nqubits = nq;
            type    = Compound;
        }

        [[nodiscard]] std::unique_ptr<Operation> clone() const override {
            std::unique_ptr<CompoundOperation> cloned_co = std::make_unique<CompoundOperation>(nqubits);
            cloned_co->reserve(ops.size());

            for (auto& op: ops) {
                cloned_co->ops.emplace_back<>(op->clone());
            }
            return cloned_co;
        }

        void setNqubits(dd::QubitCount nq) override {
            nqubits = nq;
            for (auto& op: ops) {
                op->setNqubits(nq);
            }
        }

        [[nodiscard]] bool isCompoundOperation() const override {
            return true;
        }

        [[nodiscard]] bool isNonUnitaryOperation() const override {
            return std::any_of(ops.cbegin(), ops.cend(), [](const auto& op) { return op->isNonUnitaryOperation(); });
        }

        MatrixDD getDD(std::unique_ptr<dd::Package>& dd) const override {
            MatrixDD e = dd->makeIdent(nqubits);
            for (auto& op: ops) {
                e = dd->multiply(op->getDD(dd), e);
            }
            return e;
        }

        MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd) const override {
            MatrixDD e = dd->makeIdent(nqubits);
            for (auto& op: ops) {
                e = dd->multiply(e, op->getInverseDD(dd));
            }
            return e;
        }

        MatrixDD getDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const override {
            MatrixDD e = dd->makeIdent(nqubits);
            for (auto& op: ops) {
                e = dd->multiply(op->getDD(dd, permutation), e);
            }
            return e;
        }

        MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const override {
            MatrixDD e = dd->makeIdent(nqubits);
            for (auto& op: ops) {
                e = dd->multiply(e, op->getInverseDD(dd, permutation));
            }
            return e;
        }

        std::ostream& print(std::ostream& os) const override {
            os << name;
            for (const auto& op: ops) {
                os << std::endl
                   << "\t";
                op->print(os);
            }

            return os;
        }

        std::ostream& print(std::ostream& os, const Permutation& permutation) const override {
            os << name;
            for (const auto& op: ops) {
                os << std::endl
                   << "\t";
                op->print(os, permutation);
            }

            return os;
        }

        [[nodiscard]] bool actsOn(dd::Qubit i) const override {
            return std::any_of(ops.cbegin(), ops.cend(), [&i](const auto& op) { return op->actsOn(i); });
        }

        void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg) const override {
            for (const auto& op: ops) {
                op->dumpOpenQASM(of, qreg, creg);
            }
        }

        void dumpQiskit(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg, const char* anc_reg_name) const override {
            for (const auto& op: ops) {
                op->dumpQiskit(of, qreg, creg, anc_reg_name);
            }
        }

        void dumpTensor(std::ostream& of, std::vector<std::size_t>& inds, std::size_t& gateIdx, std::unique_ptr<dd::Package>& dd) override {
            for (const auto& op: ops) {
                if (op != (*ops.begin())) {
                    of << ",\n";
                }
                op->dumpTensor(of, inds, gateIdx, dd);
            }
        }

        /**
		 * Pass-Through
		 */

        // Iterators (pass-through)
        auto               begin() noexcept { return ops.begin(); }
        [[nodiscard]] auto begin() const noexcept { return ops.begin(); }
        [[nodiscard]] auto cbegin() const noexcept { return ops.cbegin(); }
        auto               end() noexcept { return ops.end(); }
        [[nodiscard]] auto end() const noexcept { return ops.end(); }
        [[nodiscard]] auto cend() const noexcept { return ops.cend(); }
        auto               rbegin() noexcept { return ops.rbegin(); }
        [[nodiscard]] auto rbegin() const noexcept { return ops.rbegin(); }
        [[nodiscard]] auto crbegin() const noexcept { return ops.crbegin(); }
        auto               rend() noexcept { return ops.rend(); }
        [[nodiscard]] auto rend() const noexcept { return ops.rend(); }
        [[nodiscard]] auto crend() const noexcept { return ops.crend(); }

        // Capacity (pass-through)
        [[nodiscard]] bool        empty() const noexcept { return ops.empty(); }
        [[nodiscard]] std::size_t size() const noexcept { return ops.size(); }
        [[nodiscard]] std::size_t max_size() const noexcept { return ops.max_size(); }
        [[nodiscard]] std::size_t capacity() const noexcept { return ops.capacity(); }

        void reserve(std::size_t new_cap) { ops.reserve(new_cap); }
        void shrink_to_fit() { ops.shrink_to_fit(); }

        // Modifiers (pass-through)
        void                                              clear() noexcept { ops.clear(); }
        void                                              pop_back() { return ops.pop_back(); }
        void                                              resize(std::size_t count) { ops.resize(count); }
        std::vector<std::unique_ptr<Operation>>::iterator erase(std::vector<std::unique_ptr<Operation>>::const_iterator pos) { return ops.erase(pos); }
        std::vector<std::unique_ptr<Operation>>::iterator erase(std::vector<std::unique_ptr<Operation>>::const_iterator first, std::vector<std::unique_ptr<Operation>>::const_iterator last) { return ops.erase(first, last); }
        template<class T>
        void emplace_back(std::unique_ptr<T>& op) {
            ops.emplace_back(std::move(op));
        }
        template<class T, class... Args>
        void emplace_back(Args&&... args) {
            ops.emplace_back(std::make_unique<T>(args...));
        }
        template<class T, class... Args>
        std::vector<std::unique_ptr<Operation>>::iterator insert(std::vector<std::unique_ptr<Operation>>::const_iterator iterator, Args&&... args) {
            return ops.insert(iterator, std::make_unique<T>(args...));
        }
        template<class T>
        std::vector<std::unique_ptr<Operation>>::iterator insert(std::vector<std::unique_ptr<Operation>>::const_iterator iterator, std::unique_ptr<T>& op) {
            return ops.insert(iterator, std::move(op));
        }

        [[nodiscard]] const auto& at(std::size_t i) const { return ops.at(i); }

        std::vector<std::unique_ptr<Operation>>& getOps() { return ops; }
    };
} // namespace qc
#endif //QFR_COMPOUNDOPERATION_H
