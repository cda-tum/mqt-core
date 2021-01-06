/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef INTERMEDIATEREPRESENTATION_STANDARDOPERATION_H
#define INTERMEDIATEREPRESENTATION_STANDARDOPERATION_H

#include "Operation.hpp"

namespace qc {
	using GateMatrix = std::array<dd::ComplexValue, dd::NEDGE>;

	//constexpr long double PARAMETER_TOLERANCE = dd::ComplexNumbers::TOLERANCE * 10e-2;
	constexpr fp PARAMETER_TOLERANCE = 10e-6;
	inline bool fp_equals(const fp a, const fp b) { return (std::abs(a-b) <PARAMETER_TOLERANCE); }

	// Complex constants
	constexpr dd::ComplexValue complex_one        = { 1, 0 };
	constexpr dd::ComplexValue complex_mone       = { -1, 0 };
	constexpr dd::ComplexValue complex_zero       = { 0, 0 };
	constexpr dd::ComplexValue complex_i          = { 0, 1 };
	constexpr dd::ComplexValue complex_mi         = { 0, -1 };
	constexpr dd::ComplexValue complex_SQRT_2     = {  dd::SQRT_2, 0 };
	constexpr dd::ComplexValue complex_mSQRT_2    = { -dd::SQRT_2, 0 };
	constexpr dd::ComplexValue complex_iSQRT_2     = {  0, dd::SQRT_2 };
	constexpr dd::ComplexValue complex_miSQRT_2    = { 0, -dd::SQRT_2 };

	// Gate matrices
	constexpr GateMatrix Imat({    complex_one,        complex_zero,       complex_zero,       complex_one });
	constexpr GateMatrix Hmat({    complex_SQRT_2,     complex_SQRT_2,     complex_SQRT_2,     complex_mSQRT_2 });
	constexpr GateMatrix Xmat({    complex_zero,       complex_one,        complex_one,        complex_zero });
	constexpr GateMatrix Ymat({    complex_zero,       complex_mi,         complex_i,          complex_zero });
	constexpr GateMatrix Zmat({    complex_one,        complex_zero,       complex_zero,       complex_mone });
	constexpr GateMatrix Smat({    complex_one,        complex_zero,       complex_zero,       complex_i });
	constexpr GateMatrix Sdagmat({ complex_one,        complex_zero,       complex_zero,       complex_mi });
	constexpr GateMatrix Tmat({    complex_one,        complex_zero,       complex_zero,       dd::ComplexValue{ dd::SQRT_2, dd::SQRT_2 }});
	constexpr GateMatrix Tdagmat({ complex_one,        complex_zero,       complex_zero,       dd::ComplexValue{ dd::SQRT_2, -dd::SQRT_2 }});
	constexpr GateMatrix Vmat({    complex_SQRT_2,  complex_miSQRT_2, complex_miSQRT_2, complex_SQRT_2 });
	constexpr GateMatrix Vdagmat({    complex_SQRT_2,  complex_iSQRT_2, complex_iSQRT_2, complex_SQRT_2 });

	inline GateMatrix U3mat(fp lambda, fp phi, fp theta) {
		return GateMatrix({ dd::ComplexValue{  std::cos(theta / 2), 0 },
		                    dd::ComplexValue{ -std::cos(lambda)       * std::sin(theta / 2), -std::sin(lambda)      * std::sin(theta / 2) },
		                    dd::ComplexValue{  std::cos(phi)          * std::sin(theta / 2), std::sin(phi)          * std::sin(theta / 2) },
		                    dd::ComplexValue{  std::cos(lambda + phi) * std::cos(theta / 2), std::sin(lambda + phi) * std::cos(theta / 2) }});
	}

	inline GateMatrix U2mat(fp lambda, fp phi) {
		return GateMatrix({ complex_SQRT_2,
		                    dd::ComplexValue{ -std::cos(lambda)       * dd::SQRT_2, -std::sin(lambda)       * dd::SQRT_2 },
		                    dd::ComplexValue{  std::cos(phi)          * dd::SQRT_2,  std::sin(phi)          * dd::SQRT_2 },
		                    dd::ComplexValue{  std::cos(lambda + phi) * dd::SQRT_2,  std::sin(lambda + phi) * dd::SQRT_2 }});
	}

	inline GateMatrix Phasemat(fp lambda) {
		return GateMatrix({ complex_one,  complex_zero,
		                    complex_zero, dd::ComplexValue{ std::cos(lambda), std::sin(lambda) }});
	}

	constexpr GateMatrix SXmat{dd::ComplexValue{0.5, 0.5}, dd::ComplexValue{0.5, -0.5}, dd::ComplexValue{0.5, -0.5}, dd::ComplexValue{0.5, 0.5}};

	constexpr GateMatrix SXdagmat{dd::ComplexValue{0.5, -0.5}, dd::ComplexValue{0.5, 0.5}, dd::ComplexValue{0.5, 0.5}, dd::ComplexValue{0.5, -0.5}};

	inline GateMatrix RXmat(fp lambda) {
		return GateMatrix({ dd::ComplexValue{ std::cos(lambda / 2), 0 }, 
		                    dd::ComplexValue{ 0, -std::sin(lambda / 2) },
		                    dd::ComplexValue{ 0, -std::sin(lambda / 2) }, 
							dd::ComplexValue{ std::cos(lambda / 2), 0 }});
	}

	inline GateMatrix RYmat(fp lambda) {
		return GateMatrix({ dd::ComplexValue{  std::cos(lambda / 2), 0 },
		                    dd::ComplexValue{ -std::sin(lambda / 2), 0 },
		                    dd::ComplexValue{  std::sin(lambda / 2), 0 },
		                    dd::ComplexValue{  std::cos(lambda / 2), 0 }});
	}

	inline GateMatrix RZmat(fp lambda) {
		return GateMatrix({ dd::ComplexValue{ -std::cos(lambda/2.), -std::sin(lambda/2.) },  complex_zero,
		                    complex_zero, dd::ComplexValue{ std::cos(lambda/2.), std::sin(lambda/2.) }});
	}

	class StandardOperation : public Operation {
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
		StandardOperation(unsigned short nq, unsigned short                     target, OpType g, fp lambda = 0, fp phi = 0, fp theta = 0);
		StandardOperation(unsigned short nq, const std::vector<unsigned short>& targets, OpType g, fp lambda = 0, fp phi = 0, fp theta = 0);

		StandardOperation(unsigned short nq, Control control, unsigned short                     target, OpType g, fp lambda = 0, fp phi = 0, fp theta = 0);
		StandardOperation(unsigned short nq, Control control, const std::vector<unsigned short>& targets, OpType g, fp lambda = 0, fp phi = 0, fp theta = 0);

		StandardOperation(unsigned short nq, const std::vector<Control>& controls, unsigned short                     target, OpType g, fp lambda = 0, fp phi = 0, fp theta = 0);
		StandardOperation(unsigned short nq, const std::vector<Control>& controls, const std::vector<unsigned short>& targets, OpType g, fp lambda = 0, fp phi = 0, fp theta = 0);

		// MCT Constructor
		StandardOperation(unsigned short nq, const std::vector<Control>& controls, unsigned short target);

		// MCF (cSWAP), Peres, paramterized two target Constructor
		StandardOperation(unsigned short nq, const std::vector<Control>& controls, unsigned short target0, unsigned short target1, OpType g, fp lambda = 0, fp phi = 0, fp theta = 0);

		bool isStandardOperation() const override {
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

}
#endif //INTERMEDIATEREPRESENTATION_STANDARDOPERATION_H
