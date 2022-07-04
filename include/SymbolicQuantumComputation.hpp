#pragma once

#include "Expression.hpp"
#include "QuantumComputation.hpp"
#include "operations/SymbolicOperation.hpp"

#include <algorithm>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qc {

    class SymbolicQuantumComputation: public QuantumComputation {
    protected:
        std::unordered_set<sym::Variable> occuringVariables;

        void addVariables(const SymbolOrNumber& expr);
        void addVariables(const SymbolOrNumber& expr1, const SymbolOrNumber& expr2);
        void addVariables(const SymbolOrNumber& expr1, const SymbolOrNumber& expr2, const SymbolOrNumber& expr3);

    public:
        using QuantumComputation::QuantumComputation;

        void u3(dd::Qubit target, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta);
        void u3(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta);
        void u3(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta);

        void u2(dd::Qubit target, const SymbolOrNumber& lambda, const SymbolOrNumber& phi);
        void u2(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda, const SymbolOrNumber& phi);
        void u2(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda, const SymbolOrNumber& phi);

        void phase(dd::Qubit target, const SymbolOrNumber& lambda);
        void phase(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda);
        void phase(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda);

        void rx(dd::Qubit target, const SymbolOrNumber& lambda);
        void rx(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda);
        void rx(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda);

        void ry(dd::Qubit target, const SymbolOrNumber& lambda);
        void ry(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda);
        void ry(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda);

        void rz(dd::Qubit target, const SymbolOrNumber& lambda);
        void rz(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda);
        void rz(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda);

        // [[nodiscard]] QuantumComputation getInstantiatedComputation(const VariableAssignment& assignment) const;

        // Instantiates this computation
        // Afterwards casting to QuantumComputation can be done if assignment is total
        void instantiate(const VariableAssignment& assignment);

        [[nodiscard]] bool isVariableFree() const {
            return std::all_of(begin(), end(), [](const auto& op) { return op->isStandardOperation(); });
        }
    };

} // namespace qc
