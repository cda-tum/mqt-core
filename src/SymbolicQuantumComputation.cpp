#include "SymbolicQuantumComputation.hpp"

#include "Definitions.hpp"
#include "Expression.hpp"
#include "QuantumComputation.hpp"
#include "operations/StandardOperation.hpp"

#include <memory>
#include <variant>

namespace qc {

    void SymbolicQuantumComputation::addVariables(const SymbolOrNumber& expr) {
        if (std::holds_alternative<Symbolic>(expr)) {
            const auto& sym = std::get<Symbolic>(expr);
            for (const auto& term: sym)
                occuringVariables.insert(term.getVar());
        }
    }
    void SymbolicQuantumComputation::addVariables(const SymbolOrNumber& expr1, const SymbolOrNumber& expr2) {
        addVariables(expr1);
        addVariables(expr2);
    }
    void SymbolicQuantumComputation::addVariables(const SymbolOrNumber& expr1, const SymbolOrNumber& expr2, const SymbolOrNumber& expr3) {
        addVariables(expr1);
        addVariables(expr2);
        addVariables(expr3);
    }

    void SymbolicQuantumComputation::u3(dd::Qubit target, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta) {
        checkQubitRange(target);
        addVariables(lambda, phi, theta);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::U3, lambda, phi, theta);
    }
    void SymbolicQuantumComputation::u3(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta) {
        checkQubitRange(target, control);
        addVariables(lambda, phi, theta);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::U3, lambda, phi, theta);
    }
    void SymbolicQuantumComputation::u3(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta) {
        checkQubitRange(target, controls);
        addVariables(lambda, phi, theta);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::U3, lambda, phi, theta);
    }

    void SymbolicQuantumComputation::u2(dd::Qubit target, const SymbolOrNumber& lambda, const SymbolOrNumber& phi) {
        checkQubitRange(target);
        addVariables(lambda, phi);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::U2, lambda, phi);
    }
    void SymbolicQuantumComputation::u2(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda, const SymbolOrNumber& phi) {
        checkQubitRange(target, control);
        addVariables(lambda, phi);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::U2, lambda, phi);
    }
    void SymbolicQuantumComputation::u2(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda, const SymbolOrNumber& phi) {
        checkQubitRange(target, controls);
        addVariables(lambda, phi);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::U2, lambda, phi);
    }

    void SymbolicQuantumComputation::phase(dd::Qubit target, const SymbolOrNumber& lambda) {
        checkQubitRange(target);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::Phase, lambda);
    }
    void SymbolicQuantumComputation::phase(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda) {
        checkQubitRange(target, control);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::Phase, lambda);
    }
    void SymbolicQuantumComputation::phase(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda) {
        checkQubitRange(target, controls);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::Phase, lambda);
    }

    void SymbolicQuantumComputation::rx(dd::Qubit target, const SymbolOrNumber& lambda) {
        checkQubitRange(target);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::RX, lambda);
    }
    void SymbolicQuantumComputation::rx(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda) {
        checkQubitRange(target, control);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::RX, lambda);
    }
    void SymbolicQuantumComputation::rx(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda) {
        checkQubitRange(target, controls);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::RX, lambda);
    }

    void SymbolicQuantumComputation::ry(dd::Qubit target, const SymbolOrNumber& lambda) {
        checkQubitRange(target);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::RY, lambda);
    }
    void SymbolicQuantumComputation::ry(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda) {
        checkQubitRange(target, control);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::RY, lambda);
    }
    void SymbolicQuantumComputation::ry(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda) {
        checkQubitRange(target, controls);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::RY, lambda);
    }

    void SymbolicQuantumComputation::rz(dd::Qubit target, const SymbolOrNumber& lambda) {
        checkQubitRange(target);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::RZ, lambda);
    }
    void SymbolicQuantumComputation::rz(dd::Qubit target, const dd::Control& control, const SymbolOrNumber& lambda) {
        checkQubitRange(target, control);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::RZ, lambda);
    }
    void SymbolicQuantumComputation::rz(dd::Qubit target, const dd::Controls& controls, const SymbolOrNumber& lambda) {
        checkQubitRange(target, controls);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::RZ, lambda);
    }

    // QuantumComputation SymbolicQuantumComputation::getInstantiatedComputation(const VariableAssignment& assignment) const {
    //     QuantumComputation qc;
    //     for (const auto& op: *this) {
    //         const auto* symOp = dynamic_cast<SymbolicOperation*>(op.get());

    //         if (symOp) {
    //             const auto stdOp = symOp->getInstantiatedOperation(assignment);
    //             qc.emplace_back<StandardOperation>(stdOp.getNqubits(), stdOp.getControls(), stdOp.getTargets(), stdOp.getType(), stdOp.getParameter()[0], stdOp.getParameter()[1], stdOp.getParameter()[2], stdOp.getStartingQubit());
    //         } else {
    //             qc.emplace_back<Operation>(std::make_unique<Operation>(op->clone()));
    //         }
    //     }
    //     return qc;
    // }

    // Instantiates this computation
    // Afterwards casting to QuantumComputation can be done if assignment is total
    void SymbolicQuantumComputation::instantiate(const VariableAssignment& assignment) {
        for (auto& op: *this) {
            auto* symOp = dynamic_cast<SymbolicOperation*>(op.get());
            if (symOp) {
                symOp->instantiate(assignment);
            }
        }
    }

} // namespace qc
