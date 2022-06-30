#pragma once

#include "Operation.hpp"

namespace qc {
    class SymbolicOperation final: public Operation {
    protected:
        std::array<Symbolic, MAX_PARAMETERS> symbolicParameter{};

        static OpType parseU3(const Symbolic& lambda, const Symbolic& phi, const Symbolic& theta);
        static OpType parseU2(const Symbolic& lambda, const Symbolic& phi);
        static OpType parseU1(const Symbolic& lambda);

        void checkUgate();
        void setup(dd::QubitCount nq, const Symbolic& par0, const Symbolic& par1, const Symbolic& par2, dd::Qubit startingQubit = 0);

    public:
        SymbolicOperation() = default;

        [[nodiscard]] const std::array<Symbolic, MAX_PARAMETERS>& getSymbolicParameter() const {
            return symbolicParameter;
        }

        std::array<Symbolic, MAX_PARAMETERS>& getSymbolicParameter() {
            return symbolicParameter;
        }

        void setSymbolicParameter(const std::array<Symbolic, MAX_PARAMETERS>& p) {
            symbolicParameter = p;
        }

        // Standard Constructors
        SymbolicOperation(dd::QubitCount nq, dd::Qubit target, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);
        SymbolicOperation(dd::QubitCount nq, const Targets& targets, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);

        SymbolicOperation(dd::QubitCount nq, dd::Control control, dd::Qubit target, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);
        SymbolicOperation(dd::QubitCount nq, dd::Control control, const Targets& targets, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);

        SymbolicOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);
        SymbolicOperation(dd::QubitCount nq, const dd::Controls& controls, const Targets& targets, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);

        // MCT Constructor
        SymbolicOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target, dd::Qubit startingQubit = 0);

        // MCF (cSWAP), Peres, paramterized two target Constructor
        SymbolicOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target0, dd::Qubit target1, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);

        [[nodiscard]] std::unique_ptr<Operation> clone() const override {
            return std::make_unique<SymbolicOperation>(getNqubits(), getControls(), getTargets(), getType(), getParameter().at(0), getParameter().at(1), getParameter().at(2), getStartingQubit());
        }

        [[nodiscard]] inline bool isSymbolicOperation() const override {
            return true;
        }
    };
} // namespace qc
