#pragma once

#include "Definitions.hpp"
#include "Expression.hpp"
#include "StandardOperation.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <unordered_set>
#include <variant>

namespace qc {
    // Overload pattern for std::visit
    template<typename... Ts>
    struct Overload: Ts... {
        using Ts::operator()...;
    };
    template<class... Ts>
    Overload(Ts...) -> Overload<Ts...>;

    class SymbolicOperation final: public StandardOperation {
    protected:
        std::array<std::optional<Symbolic>, MAX_PARAMETERS> symbolicParameter{{std::nullopt, std::nullopt, std::nullopt}};

        static OpType parseU3(const Symbolic& lambda, dd::fp& phi, dd::fp& theta);
        static OpType parseU3(dd::fp& lambda, const Symbolic& phi, dd::fp& theta);
        static OpType parseU3(dd::fp& lambda, dd::fp& phi, const Symbolic& theta);
        static OpType parseU3(const Symbolic& lambda, const Symbolic& phi, dd::fp& theta);
        static OpType parseU3(const Symbolic& lambda, dd::fp& phi, const Symbolic& theta);
        static OpType parseU3(dd::fp& lambda, const Symbolic& phi, const Symbolic& theta);

        static OpType parseU2(const Symbolic& lambda, const Symbolic& phi);
        static OpType parseU2(const Symbolic& lambda, dd::fp& phi);
        static OpType parseU2(dd::fp& lambda, const Symbolic& phi);

        static OpType parseU1(const Symbolic& lambda);

        void checkUgate();

        void storeSymbolOrNumber(const SymbolOrNumber& param, std::size_t i);

        [[nodiscard]] bool isSymbolicParameter(std::size_t i) const {
            return symbolicParameter[i].has_value();
        }

        static bool isSymbol(const SymbolOrNumber& param) {
            return std::holds_alternative<Symbolic>(param);
        }

        static Symbolic& getSymbol(SymbolOrNumber& param) {
            return std::get<Symbolic>(param);
        }

        static dd::fp& getNumber(SymbolOrNumber& param) {
            return std::get<dd::fp>(param);
        }

        void setup(dd::QubitCount nq, const SymbolOrNumber& par0, const SymbolOrNumber& par1, const SymbolOrNumber& par2, dd::Qubit startingQubit = 0);

        [[nodiscard]] static dd::fp getInstantiation(const SymbolOrNumber& symOrNum, const VariableAssignment& assignment);

    public:
        SymbolicOperation() = default;

        // [[nodiscard]] const std::array<Symbolic, MAX_PARAMETERS>& getSymbolicParameter() const {
        //     return symbolicParameter;
        // }

        // std::array<Symbolic, MAX_PARAMETERS>& getSymbolicParameter() {
        //     return symbolicParameter;
        // }

        [[nodiscard]] SymbolOrNumber getParameter(std::size_t i) const {
            if (symbolicParameter[i].has_value()) return symbolicParameter[i].value();
            return parameter[i];
        }

        void setSymbolicParameter(const Symbolic& par, std::size_t i) {
            symbolicParameter[i] = par;
        }

        // Standard Constructors
        SymbolicOperation(dd::QubitCount nq, dd::Qubit target, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, dd::Qubit startingQubit = 0);
        SymbolicOperation(dd::QubitCount nq, const Targets& targets, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, dd::Qubit startingQubit = 0);

        SymbolicOperation(dd::QubitCount nq, dd::Control control, dd::Qubit target, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, dd::Qubit startingQubit = 0);
        SymbolicOperation(dd::QubitCount nq, dd::Control control, const Targets& targets, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, dd::Qubit startingQubit = 0);

        SymbolicOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, dd::Qubit startingQubit = 0);
        SymbolicOperation(dd::QubitCount nq, const dd::Controls& controls, const Targets& targets, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, dd::Qubit startingQubit = 0);

        // MCF (cSWAP), Peres, paramterized two target Constructor
        SymbolicOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target0, dd::Qubit target1, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, dd::Qubit startingQubit = 0);

        [[nodiscard]] std::unique_ptr<Operation> clone() const override {
            return std::make_unique<SymbolicOperation>(getNqubits(), getControls(), getTargets(), getType(), getParameter(0), getParameter(1), getParameter(2), getStartingQubit());
        }

        [[nodiscard]] inline bool isSymbolicOperation() const override {
            return true;
        }

        [[nodiscard]] inline bool isStandardOperation() const override {
            return std::all_of(symbolicParameter.begin(), symbolicParameter.end(), [](const auto& sym) { return !sym.has_value(); });
        }

        // [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1, const Permutation& perm2) const override {
        //     return Operation::equals(op, perm1, perm2);
        // }
        // [[nodiscard]] bool equals(const Operation& operation) const override {
        //     return equals(operation, {}, {});
        // }

        [[nodiscard]] bool        equals(const Operation& op, const Permutation& perm1, const Permutation& perm2) const override;
        [[nodiscard]] inline bool equals(const Operation& op) const override {
            return equals(op, {}, {});
        }

        void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg) const override;
        void dumpQiskit(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg, const char* anc_reg_name) const override;

        [[nodiscard]] StandardOperation getInstantiatedOperation(const VariableAssignment& assignment) const;

        // Instantiates this Operation
        // Afterwards casting to StandardOperation can be done if assignment is total
        void instantiate(const VariableAssignment& assignment);
    };
} // namespace qc
