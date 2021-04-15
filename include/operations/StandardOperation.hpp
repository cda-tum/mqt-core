/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_STANDARDOPERATION_H
#define QFR_STANDARDOPERATION_H

#include "DDpackage.h"
#include "GateMatrixDefinitions.h"
#include "Operation.hpp"

namespace qc {
    using GateMatrix = std::array<dd::ComplexValue, dd::NEDGE>;

    //constexpr long double PARAMETER_TOLERANCE = dd::ComplexNumbers::TOLERANCE * 10e-2;
    constexpr fp PARAMETER_TOLERANCE = 10e-6;
    inline bool  fp_equals(const fp a, const fp b) { return (std::abs(a - b) < PARAMETER_TOLERANCE); }

    class StandardOperation: public Operation {
    protected:
        static void checkInteger(fp& ld) {
            auto nearest = std::nearbyint(ld);
            if (std::abs(ld - nearest) < PARAMETER_TOLERANCE) {
                ld = nearest;
            }
        }

        static void checkFractionPi(fp& ld) {
            auto div     = qc::PI / ld;
            auto nearest = std::nearbyint(div);
            if (std::abs(div - nearest) < PARAMETER_TOLERANCE) {
                ld = qc::PI / nearest;
            }
        }

        static OpType parseU3(fp& lambda, fp& phi, fp& theta);
        static OpType parseU2(fp& lambda, fp& phi);
        static OpType parseU1(fp& lambda);

        void checkUgate();
        void setup(unsigned short nq, fp par0, fp par1, fp par2);

        dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, bool inverse, const std::map<unsigned short, unsigned short>& permutation = standardPermutation) const;

    public:
        StandardOperation() = default;

        // Standard Constructors
        StandardOperation(unsigned short nq, unsigned short target, OpType g, fp lambda = 0., fp phi = 0., fp theta = 0.);
        StandardOperation(unsigned short nq, const std::vector<unsigned short>& targets, OpType g, fp lambda = 0., fp phi = 0., fp theta = 0.);

        StandardOperation(unsigned short nq, Control control, unsigned short target, OpType g, fp lambda = 0., fp phi = 0., fp theta = 0.);
        StandardOperation(unsigned short nq, Control control, const std::vector<unsigned short>& targets, OpType g, fp lambda = 0., fp phi = 0., fp theta = 0.);

        StandardOperation(unsigned short nq, const std::vector<Control>& controls, unsigned short target, OpType g, fp lambda = 0., fp phi = 0., fp theta = 0.);
        StandardOperation(unsigned short nq, const std::vector<Control>& controls, const std::vector<unsigned short>& targets, OpType g, fp lambda = 0., fp phi = 0., fp theta = 0.);

        // MCT Constructor
        StandardOperation(unsigned short nq, const std::vector<Control>& controls, unsigned short target);

        // MCF (cSWAP), Peres, paramterized two target Constructor
        StandardOperation(unsigned short nq, const std::vector<Control>& controls, unsigned short target0, unsigned short target1, OpType g, fp lambda = 0., fp phi = 0., fp theta = 0.);

        [[nodiscard]] bool isStandardOperation() const override {
            return true;
        }

        dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const override;
        dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const override;

        dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const override;
        dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const override;

        dd::Edge getSWAPDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation = standardPermutation) const;
        dd::Edge getPeresDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation = standardPermutation) const;
        dd::Edge getPeresdagDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation = standardPermutation) const;
        dd::Edge getiSWAPDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation = standardPermutation) const;
        dd::Edge getiSWAPinvDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation = standardPermutation) const;

        void dumpOpenQASM(std::ostream& of, const regnames_t& qreg, const regnames_t& creg) const override;
        void dumpReal(std::ostream& of) const override;
        void dumpQiskit(std::ostream& of, const regnames_t& qreg, const regnames_t& creg, const char* anc_reg_name) const override;
    };

} // namespace qc
#endif //QFR_STANDARDOPERATION_H
