//
// Created by Lukas Burgholzer on 27.09.19.
//

#ifndef INTERMEDIATEREPRESENTATION_STANDARDOPERATION_H
#define INTERMEDIATEREPRESENTATION_STANDARDOPERATION_H

#include <vector>
#include <iostream>
#include <cstring>

#include "Operation.h"

typedef std::array<dd::ComplexValue,dd::NEDGE> GateMatrix;

// Complex constants
constexpr dd::ComplexValue complex_one = { 1, 0 };
constexpr dd::ComplexValue complex_mone = { -1, 0 };
constexpr dd::ComplexValue complex_zero = { 0, 0 };
constexpr dd::ComplexValue complex_one_one_2 = { 0.5, 0.5 };
constexpr dd::ComplexValue complex_one_mone_2 = { 0.5, -0.5 };
constexpr dd::ComplexValue complex_i = { 0, 1 };
constexpr dd::ComplexValue complex_mi = { 0, -1 };
constexpr dd::ComplexValue complex_SQRT_2 = { dd::SQRT_2, 0 };
constexpr dd::ComplexValue complex_mSQRT_2 = { -dd::SQRT_2, 0 };

// Gate matrices
constexpr GateMatrix Imat( { complex_one, complex_zero, complex_zero, complex_one });
constexpr GateMatrix Hmat( { complex_SQRT_2, complex_SQRT_2, complex_SQRT_2, complex_mSQRT_2 });
constexpr GateMatrix Xmat( { complex_zero, complex_one, complex_one, complex_zero });
constexpr GateMatrix Ymat( { complex_zero, complex_mi, complex_i, complex_zero });
constexpr GateMatrix Zmat( { complex_one, complex_zero, complex_zero, complex_mone });
constexpr GateMatrix Smat( { complex_one, complex_zero, complex_zero, complex_i });
constexpr GateMatrix Sdagmat( { complex_one, complex_zero, complex_zero, complex_mi });
constexpr GateMatrix Tmat( { complex_one, complex_zero, complex_zero, dd::ComplexValue{ dd::SQRT_2, dd::SQRT_2 }});
constexpr GateMatrix Tdagmat( { complex_one, complex_zero, complex_zero, dd::ComplexValue{ dd::SQRT_2, -dd::SQRT_2 }});
constexpr GateMatrix Vmat( { complex_one_one_2, complex_one_mone_2, complex_one_mone_2, complex_one_one_2 });
constexpr GateMatrix Vdagmat( { complex_one_mone_2, complex_one_one_2, complex_one_one_2, complex_one_mone_2 });

inline GateMatrix U3mat(long double lambda, long double phi, long double theta) {
	return GateMatrix({ dd::ComplexValue{ std::cos(theta / 2), 0 },
	         dd::ComplexValue{ -std::cos(lambda) * std::sin(theta / 2), -std::sin(lambda) * std::sin(theta / 2) },
	         dd::ComplexValue{ std::cos(phi) * std::sin(theta / 2), std::sin(phi) * std::sin(theta / 2) },
	         dd::ComplexValue{ std::cos(lambda + phi) * std::cos(theta / 2), std::sin(lambda + phi) * std::cos(theta / 2) }});
}
inline GateMatrix U3dagmat(long double lambda, long double phi, long double theta) {
	return GateMatrix({ dd::ComplexValue{ std::cos(theta / 2), 0 },
	         dd::ComplexValue{ std::cos(phi) * std::sin(theta / 2), -std::sin(phi) * std::sin(theta / 2) },
	         dd::ComplexValue{ -std::cos(lambda) * std::sin(theta / 2), std::sin(lambda) * std::sin(theta / 2) },
	         dd::ComplexValue{ std::cos(lambda + phi) * std::cos(theta / 2), -std::sin(lambda + phi) * std::cos(theta / 2) }});
}
inline GateMatrix U2mat(long double lambda, long double phi) {
	return GateMatrix({ complex_SQRT_2,
	         dd::ComplexValue{ -std::cos(lambda) * dd::SQRT_2, -std::sin(lambda) * dd::SQRT_2 },
	         dd::ComplexValue{ std::cos(phi) * dd::SQRT_2, std::sin(phi) * dd::SQRT_2},
	         dd::ComplexValue{ std::cos(lambda + phi) * dd::SQRT_2, std::sin(lambda + phi) * dd::SQRT_2 }});
}
inline GateMatrix U2dagmat(long double lambda, long double phi) {
	return GateMatrix({ complex_SQRT_2,
	         dd::ComplexValue{ std::cos(phi) * dd::SQRT_2, -std::sin(phi) * dd::SQRT_2 },
	         dd::ComplexValue{ -std::cos(lambda) * dd::SQRT_2 , std::sin(lambda) * dd::SQRT_2 },
	         dd::ComplexValue{ std::cos(lambda + phi) * dd::SQRT_2, -std::sin(lambda + phi) * dd::SQRT_2 }});
}
inline GateMatrix RXmat(long double lambda) {
	return GateMatrix({ dd::ComplexValue{ std::cos(lambda / 2), 0 }, dd::ComplexValue{ 0, -std::sin(lambda / 2) },
	         dd::ComplexValue{ 0, -std::sin(lambda / 2) }, dd::ComplexValue{ std::cos(lambda / 2), 0 }});
}
inline GateMatrix RYmat(long double lambda) {
	return GateMatrix({ dd::ComplexValue{ std::cos(lambda / 2), 0 },
	         dd::ComplexValue{ -std::sin(lambda / 2), 0 },
	         dd::ComplexValue{ std::sin(lambda / 2), 0 },
	         dd::ComplexValue{ std::cos(lambda / 2), 0 }});
}
inline GateMatrix RZmat(long double lambda) {
	return GateMatrix({ complex_one, complex_zero,
	         complex_zero, dd::ComplexValue{ std::cos(lambda), std::sin(lambda) }});
}

// Supported Operations
enum Gate: short {I, H, X, Y, Z, S, Sdag, T, Tdag, V, Vdag, U3, U2, U1, RX, RY, RZ, SWAP, P, Pdag, Measure};

class StandardOperation : public Operation {
protected:
	const Gate gate = I; // Gate type
public:
	StandardOperation() = default;
	// Standard Constructor
	StandardOperation(short nq, const std::vector<short>& controls, const std::vector<short>& targets, Gate g, double lambda = 0, double phi = 0, double theta = 0);
	// MCT Constructor
	StandardOperation(short nq, const std::vector<short>& controls, short target): StandardOperation(nq,controls,{target},X) {
		parameter[0] = 1;
		parameter[1] = controls.size();
		parameter[2] = target;
	}
	// MCF (cSWAP) and Peres Constructor
	StandardOperation(short nq, const std::vector<short>& controls, short target0, short target1, Gate g) : StandardOperation(nq,controls, {target0,target1},g) {
		parameter[0] = target0;
		parameter[1] = target1;
	}
	// Measurement Constructor
	StandardOperation(short nq, const std::vector<short>& qubitRegister, const std::vector<short>& classicalRegister):gate(Measure) {
		nqubits = nq;
		strcpy(name, "Measure");

		for (int i = 0; i < nqubits; ++i) {
			line[i] = -1;
		}

		// i-th qubit to be measured shall be measured into i-th classical register
		for (unsigned long i = 0; i < qubitRegister.size(); ++i) {
			line[qubitRegister[i]] = classicalRegister[i];
		}

		parameter[0] = qubitRegister.size();
	}

	dd::Edge getDD(dd::Package* dd) override;
	dd::Edge getInverseDD(dd::Package *dd) override;

	inline bool isMeasurement() const override { return gate == Measure; };
};

#endif //INTERMEDIATEREPRESENTATION_STANDARDOPERATION_H
