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
        std::array<std::optional<Symbolic>, MAX_PARAMETERS> symbolicParameter{};

        static OpType parseU3(const Symbolic& lambda, fp& phi, fp& theta);
        static OpType parseU3(fp& lambda, const Symbolic& phi, fp& theta);
        static OpType parseU3(fp& lambda, fp& phi, const Symbolic& theta);
        static OpType parseU3(const Symbolic& lambda, const Symbolic& phi, fp& theta);
        static OpType parseU3(const Symbolic& lambda, fp& phi, const Symbolic& theta);
        static OpType parseU3(fp& lambda, const Symbolic& phi, const Symbolic& theta);

        [[gnu::const]] static OpType parseU2(const Symbolic& lambda, const Symbolic& phi);
        static OpType                parseU2(const Symbolic& lambda, fp& phi);
        static OpType                parseU2(fp& lambda, const Symbolic& phi);

        [[gnu::const]] static OpType parseU1(const Symbolic& lambda);

        void checkSymbolicUgate();

        void storeSymbolOrNumber(const SymbolOrNumber& param, std::size_t i);

        [[nodiscard]] bool isSymbolicParameter(const std::size_t i) const {
            return symbolicParameter.at(i).has_value();
        }

        static bool isSymbol(const SymbolOrNumber& param) {
            return std::holds_alternative<Symbolic>(param);
        }

        static Symbolic& getSymbol(SymbolOrNumber& param) {
            return std::get<Symbolic>(param);
        }

        static fp& getNumber(SymbolOrNumber& param) {
            return std::get<fp>(param);
        }

        void setup(std::size_t nq, const SymbolOrNumber& par0, const SymbolOrNumber& par1, const SymbolOrNumber& par2, Qubit startingQubit = 0);

        [[nodiscard]] static fp getInstantiation(const SymbolOrNumber& symOrNum, const VariableAssignment& assignment);

    public:
        SymbolicOperation() = default;

        [[nodiscard]] SymbolOrNumber getParameter(const std::size_t i) const {
            if (symbolicParameter.at(i).has_value()) {
                return symbolicParameter.at(i).value();
            }
            return parameter.at(i);
        }

        void setSymbolicParameter(const Symbolic& par, const std::size_t i) {
            symbolicParameter.at(i) = par;
        }

        // Standard Constructors
        SymbolicOperation(std::size_t nq, Qubit target, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, Qubit startingQubit = 0);
        SymbolicOperation(std::size_t nq, const Targets& targ, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, Qubit startingQubit = 0);

        SymbolicOperation(std::size_t nq, Control control, Qubit target, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, Qubit startingQubit = 0);
        SymbolicOperation(std::size_t nq, Control control, const Targets& targ, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, Qubit startingQubit = 0);

        SymbolicOperation(std::size_t nq, const Controls& c, Qubit target, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, Qubit startingQubit = 0);
        SymbolicOperation(std::size_t nq, const Controls& c, const Targets& targ, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, Qubit startingQubit = 0);

        // MCF (cSWAP), Peres, paramterized two target Constructor
        SymbolicOperation(std::size_t nq, const Controls& c, Qubit target0, Qubit target1, OpType g, const SymbolOrNumber& lambda = 0.0, const SymbolOrNumber& phi = 0.0, const SymbolOrNumber& theta = 0.0, Qubit startingQubit = 0);

        [[nodiscard]] std::unique_ptr<Operation> clone() const override {
            return std::make_unique<SymbolicOperation>(getNqubits(), getControls(), getTargets(), getType(), getParameter(0), getParameter(1), getParameter(2), getStartingQubit());
        }

        [[nodiscard]] inline bool isSymbolicOperation() const override {
            return true;
        }

        [[nodiscard]] inline bool isStandardOperation() const override {
            return std::all_of(symbolicParameter.begin(), symbolicParameter.end(), [](const auto& sym) { return !sym.has_value(); });
        }

        [[nodiscard]] bool        equals(const Operation& op, const Permutation& perm1, const Permutation& perm2) const override;
        [[nodiscard]] inline bool equals(const Operation& op) const override {
            return equals(op, {}, {});
        }

        [[noreturn]] void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg) const override;

        [[nodiscard]] StandardOperation getInstantiatedOperation(const VariableAssignment& assignment) const;

        // Instantiates this Operation
        // Afterwards casting to StandardOperation can be done if assignment is total
        void instantiate(const VariableAssignment& assignment);
    };
} // namespace qc
