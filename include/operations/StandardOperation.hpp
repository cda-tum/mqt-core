/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "Operation.hpp"

namespace qc {
    class StandardOperation: public Operation {
    protected:
        static void checkInteger(dd::fp& ld) {
            dd::fp nearest = std::nearbyint(ld);
            if (std::abs(ld - nearest) < PARAMETER_TOLERANCE) {
                ld = nearest;
            }
        }

        static void checkFractionPi(dd::fp& ld) {
            dd::fp div     = dd::PI / ld;
            dd::fp nearest = std::nearbyint(div);
            if (std::abs(div - nearest) < PARAMETER_TOLERANCE) {
                ld = dd::PI / nearest;
            }
        }

        static OpType parseU3(dd::fp& lambda, dd::fp& phi, dd::fp& theta);
        static OpType parseU2(dd::fp& lambda, dd::fp& phi);
        static OpType parseU1(dd::fp& lambda);

        void checkUgate();
        void setup(dd::QubitCount nq, dd::fp par0, dd::fp par1, dd::fp par2, dd::Qubit startingQubit = 0);

        void dumpOpenQASMSwap(std::ostream& of, const RegisterNames& qreg) const;
        void dumpOpenQASMiSwap(std::ostream& of, const RegisterNames& qreg) const;
        void dumpOpenQASMTeleportation(std::ostream& of, const RegisterNames& qreg) const;

    public:
        StandardOperation() = default;

        // Standard Constructors
        StandardOperation(dd::QubitCount nq, dd::Qubit target, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);
        StandardOperation(dd::QubitCount nq, const Targets& targets, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);

        StandardOperation(dd::QubitCount nq, dd::Control control, dd::Qubit target, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);
        StandardOperation(dd::QubitCount nq, dd::Control control, const Targets& targets, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);

        StandardOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);
        StandardOperation(dd::QubitCount nq, const dd::Controls& controls, const Targets& targets, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);

        // MCT Constructor
        StandardOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target, dd::Qubit startingQubit = 0);

        // MCF (cSWAP), Peres, paramterized two target Constructor
        StandardOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target0, dd::Qubit target1, OpType g, dd::fp lambda = 0., dd::fp phi = 0., dd::fp theta = 0., dd::Qubit startingQubit = 0);

        [[nodiscard]] std::unique_ptr<Operation> clone() const override {
            return std::make_unique<StandardOperation>(getNqubits(), getControls(), getTargets(), getType(), getParameter().at(0), getParameter().at(1), getParameter().at(2), getStartingQubit());
        }

        [[nodiscard]] bool isStandardOperation() const override {
            return true;
        }

        [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1, const Permutation& perm2) const override {
            return Operation::equals(op, perm1, perm2);
        }
        [[nodiscard]] bool equals(const Operation& operation) const override {
            return equals(operation, {}, {});
        }

        void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg) const override;
    };

} // namespace qc
