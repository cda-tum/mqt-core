/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_STANDARDOPERATION_H
#define QFR_STANDARDOPERATION_H

#include "Operation.hpp"
#include "dd/GateMatrixDefinitions.hpp"

namespace qc {
    class StandardOperation final: public Operation {
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

        // single-target operations
        MatrixDD getStandardOperationDD(std::unique_ptr<dd::Package>& dd, const dd::Controls& controls, dd::Qubit target, bool inverse) const;
        // two-target operations
        MatrixDD getStandardOperationDD(std::unique_ptr<dd::Package>& dd, const dd::Controls& controls, dd::Qubit target0, dd::Qubit target1, bool inverse) const;

        MatrixDD getDD(std::unique_ptr<dd::Package>& dd, const dd::Controls& controls, const Targets& targets) const override {
            if (type == SWAP || type == iSWAP || type == Peres || type == Peresdag) {
                assert(targets.size() == 2);
                return getStandardOperationDD(dd, controls, targets[0], targets[1], false);
            } else {
                assert(targets.size() == 1);
                return getStandardOperationDD(dd, controls, targets[0], false);
            }
        }
        MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd, const dd::Controls& controls, const Targets& targets) const override {
            if (type == SWAP || type == iSWAP || type == Peres || type == Peresdag) {
                assert(targets.size() == 2);
                return getStandardOperationDD(dd, controls, targets[0], targets[1], true);
            } else {
                assert(targets.size() == 1);
                return getStandardOperationDD(dd, controls, targets[0], true);
            }
        }

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

        [[nodiscard]] bool isStandardOperation() const override {
            return true;
        }

        MatrixDD getDD(std::unique_ptr<dd::Package>& dd) const override { return Operation::getDD(dd); }
        MatrixDD getDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const override;
        MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd) const override { return Operation::getInverseDD(dd); }
        MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const override;

        void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg) const override;
        void dumpQiskit(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg, const char* anc_reg_name) const override;
    };

} // namespace qc
#endif //QFR_STANDARDOPERATION_H
