//
// Created by Lukas Burgholzer on 27.09.19.
//

#ifndef INTERMEDIATEREPRESENTATION_STANDARDOPERATION_H
#define INTERMEDIATEREPRESENTATION_STANDARDOPERATION_H

#include <vector>
#include <iostream>
#include <cstring>
#include <memory>
#include <map>

#include "Operation.h"

namespace qc {

	using GateMatrix = std::array<dd::ComplexValue, dd::NEDGE>;

	//constexpr long double PARAMETER_TOLERANCE = dd::ComplexNumbers::TOLERANCE * 10e-2;
	constexpr fp PARAMETER_TOLERANCE = 10e-6;
	inline bool fp_equals(const fp a, const fp b) { return (std::abs(a-b) <PARAMETER_TOLERANCE); }

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
	constexpr GateMatrix Imat({ complex_one, complex_zero, complex_zero, complex_one });
	constexpr GateMatrix Hmat({ complex_SQRT_2, complex_SQRT_2, complex_SQRT_2, complex_mSQRT_2 });
	constexpr GateMatrix Xmat({ complex_zero, complex_one, complex_one, complex_zero });
	constexpr GateMatrix Ymat({ complex_zero, complex_mi, complex_i, complex_zero });
	constexpr GateMatrix Zmat({ complex_one, complex_zero, complex_zero, complex_mone });
	constexpr GateMatrix Smat({ complex_one, complex_zero, complex_zero, complex_i });
	constexpr GateMatrix Sdagmat({ complex_one, complex_zero, complex_zero, complex_mi });
	constexpr GateMatrix Tmat({ complex_one, complex_zero, complex_zero, dd::ComplexValue{ dd::SQRT_2, dd::SQRT_2 }});
	constexpr GateMatrix Tdagmat({ complex_one, complex_zero, complex_zero, dd::ComplexValue{ dd::SQRT_2, -dd::SQRT_2 }});
	constexpr GateMatrix Vmat({ complex_one_one_2, complex_one_mone_2, complex_one_mone_2, complex_one_one_2 });
	constexpr GateMatrix Vdagmat({ complex_one_mone_2, complex_one_one_2, complex_one_one_2, complex_one_mone_2 });

	inline GateMatrix U3mat(fp lambda, fp phi, fp theta) {
		return GateMatrix({ dd::ComplexValue{ std::cos(theta / 2), 0 },
		                    dd::ComplexValue{ -std::cos(lambda) * std::sin(theta / 2), -std::sin(lambda) * std::sin(theta / 2) },
		                    dd::ComplexValue{ std::cos(phi) * std::sin(theta / 2), std::sin(phi) * std::sin(theta / 2) },
		                    dd::ComplexValue{ std::cos(lambda + phi) * std::cos(theta / 2), std::sin(lambda + phi) * std::cos(theta / 2) }});
	}

	inline GateMatrix U3dagmat(fp lambda, fp phi, fp theta) {
		return GateMatrix({ dd::ComplexValue{ std::cos(theta / 2), 0 },
		                    dd::ComplexValue{ std::cos(phi) * std::sin(theta / 2), -std::sin(phi) * std::sin(theta / 2) },
		                    dd::ComplexValue{ -std::cos(lambda) * std::sin(theta / 2), std::sin(lambda) * std::sin(theta / 2) },
		                    dd::ComplexValue{ std::cos(lambda + phi) * std::cos(theta / 2), -std::sin(lambda + phi) * std::cos(theta / 2) }});
	}

	inline GateMatrix U2mat(fp lambda, fp phi) {
		return GateMatrix({ complex_SQRT_2,
		                    dd::ComplexValue{ -std::cos(lambda) * dd::SQRT_2, -std::sin(lambda) * dd::SQRT_2 },
		                    dd::ComplexValue{ std::cos(phi) * dd::SQRT_2, std::sin(phi) * dd::SQRT_2 },
		                    dd::ComplexValue{ std::cos(lambda + phi) * dd::SQRT_2, std::sin(lambda + phi) * dd::SQRT_2 }});
	}

	inline GateMatrix U2dagmat(fp lambda, fp phi) {
		return GateMatrix({ complex_SQRT_2,
		                    dd::ComplexValue{ std::cos(phi) * dd::SQRT_2, -std::sin(phi) * dd::SQRT_2 },
		                    dd::ComplexValue{ -std::cos(lambda) * dd::SQRT_2, std::sin(lambda) * dd::SQRT_2 },
		                    dd::ComplexValue{ std::cos(lambda + phi) * dd::SQRT_2, -std::sin(lambda + phi) * dd::SQRT_2 }});
	}

	inline GateMatrix RXmat(fp lambda) {
		return GateMatrix({ dd::ComplexValue{ std::cos(lambda / 2), 0 }, dd::ComplexValue{ 0, -std::sin(lambda / 2) },
		                    dd::ComplexValue{ 0, -std::sin(lambda / 2) }, dd::ComplexValue{ std::cos(lambda / 2), 0 }});
	}

	inline GateMatrix RYmat(fp lambda) {
		return GateMatrix({ dd::ComplexValue{ std::cos(lambda / 2), 0 },
		                    dd::ComplexValue{ -std::sin(lambda / 2), 0 },
		                    dd::ComplexValue{ std::sin(lambda / 2), 0 },
		                    dd::ComplexValue{ std::cos(lambda / 2), 0 }});
	}

	inline GateMatrix RZmat(fp lambda) {
		return GateMatrix({ complex_one, complex_zero,
		                    complex_zero, dd::ComplexValue{ std::cos(lambda), std::sin(lambda) }});
	}

	// Supported Operations
	enum Gate : short {
		None, I, H, X, Y, Z, S, Sdag, T, Tdag, V, Vdag, U3, U2, U1, RX, RY, RZ, SWAP, P, Pdag
	};

	static const std::map<std::string, Gate> identifierMap{
			{ "0",   I },
			{ "id",  I },
			{ "h",   H },
			{ "n",   X },
			{ "c",   X },
			{ "x",   X },
			{ "y",   Y },
			{ "z",   Z },
			{ "s",   S },
			{ "si",  Sdag },
			{ "sp",  Sdag },
			{ "s+",  Sdag },
			{ "sdg", Sdag },
			{ "v",   V },
			{ "vi",  Vdag },
			{ "vp",  Vdag },
			{ "v+",  Vdag },
			{ "rx",  RX },
			{ "ry",  RY },
			{ "rz",  RZ },
			{ "f",   SWAP },
			{ "p",   P },
			{ "pi",  Pdag },
			{ "p+",  Pdag },
			{ "q",   RZ },
			{ "t",   T },
			{ "tdg", Tdag }};

	class StandardOperation : public Operation {
	protected:
		Gate gate = None; // Gate type

		static void checkInteger(fp& ld) {
			auto nearest = std::nearbyint(ld);
			if (std::abs(ld - nearest) < PARAMETER_TOLERANCE) {
				ld = nearest;
			}
		}

		static void checkFractionPi(fp& ld) {
			auto div = qc::PI / ld;
			auto nearest = std::nearbyint(div);
			if (std::abs(div - nearest) < PARAMETER_TOLERANCE) {
				ld = qc::PI / nearest;
			}
		}

		static Gate parseU3(fp& lambda, fp& phi, fp& theta) {
			if (std::abs(theta) < PARAMETER_TOLERANCE && std::abs(phi) < PARAMETER_TOLERANCE) {
				phi = theta = 0.L;
				return parseU1(lambda);
			}

			if (std::abs(theta - qc::PI_2) < PARAMETER_TOLERANCE) {
				theta = qc::PI_2;
				auto res = parseU2(lambda, phi);
				if (res != U2)
					theta = 0.L;
				return res;
			}

			if (std::abs(lambda) < PARAMETER_TOLERANCE) {
				lambda = 0.L;
				if (std::abs(phi) < PARAMETER_TOLERANCE) {
					phi = 0.L;
					checkInteger(theta);
					checkFractionPi(theta);
					lambda = theta;
					theta = 0.L;
					return RY;
				}
			}

			if (std::abs(lambda - qc::PI_2) < PARAMETER_TOLERANCE) {
				lambda = qc::PI_2;
				if (std::abs(phi + qc::PI_2) < PARAMETER_TOLERANCE) {
					phi = 0.L;
					checkInteger(theta);
					checkFractionPi(theta);
					lambda = theta;
					theta = 0.L;
					return RX;
				}

				if (std::abs(phi - qc::PI_2) < PARAMETER_TOLERANCE) {
					phi = qc::PI_2;
					if (std::abs(theta - qc::PI) < PARAMETER_TOLERANCE) {
						lambda = phi = theta = 0.L;
						return Y;
					}
				}
			}

			if (std::abs(lambda - qc::PI) < PARAMETER_TOLERANCE) {
				lambda = qc::PI;
				if (std::abs(phi) < PARAMETER_TOLERANCE) {
					phi = 0.L;
					if (std::abs(theta - qc::PI) < PARAMETER_TOLERANCE) {
						theta = lambda = 0.L;
						return X;
					}
				}
			}

			// parse a real u3 gate
			checkInteger(lambda);
			checkFractionPi(lambda);
			checkInteger(phi);
			checkFractionPi(phi);
			checkInteger(theta);
			checkFractionPi(theta);

			return U3;
		}

		static Gate parseU2(fp& lambda, fp& phi) {
			if (std::abs(phi) < PARAMETER_TOLERANCE) {
				phi = 0.L;
				if (std::abs(std::abs(lambda) - qc::PI) < PARAMETER_TOLERANCE) {
					lambda = 0.L;
					return H;
				}
				if (std::abs(lambda) < PARAMETER_TOLERANCE) {
					lambda = qc::PI_2;
					return RY;
				}
			}

			if (std::abs(lambda - qc::PI_2) < PARAMETER_TOLERANCE) {
				lambda = qc::PI_2;
				if (std::abs(phi + qc::PI_2) < PARAMETER_TOLERANCE) {
					phi = 0.L;
					return RX;
				}
			}

			checkInteger(lambda);
			checkFractionPi(lambda);
			checkInteger(phi);
			checkFractionPi(phi);

			return U2;
		}

		static Gate parseU1(fp& lambda) {
			if (std::abs(lambda) < PARAMETER_TOLERANCE) {
				lambda = 0.L;
				return I;
			}
			bool sign = std::signbit(lambda);

			if (std::abs(std::abs(lambda) - qc::PI) < PARAMETER_TOLERANCE) {
				lambda = 0.L;
				return Z;
			}

			if (std::abs(std::abs(lambda) - qc::PI_2) < PARAMETER_TOLERANCE) {
				lambda = 0.L;
				return sign ? Sdag : S;
			}

			if (std::abs(std::abs(lambda) - qc::PI_4) < PARAMETER_TOLERANCE) {
				lambda = 0.L;
				return sign ? Tdag : T;
			}

			checkInteger(lambda);
			checkFractionPi(lambda);

			return RZ;
		}

		void setName() {
			switch (gate) {
				case I: strcpy(name, "I   ");
					break;
				case H: strcpy(name, "H   ");
					break;
				case X: strcpy(name, "X   ");
					break;
				case Y: strcpy(name, "Y   ");
					break;
				case Z: strcpy(name, "Z   ");
					break;
				case S: strcpy(name, "S   ");
					break;
				case Sdag: strcpy(name, "Sdag");
					break;
				case T: strcpy(name, "T   ");
					break;
				case Tdag: strcpy(name, "Tdag");
					break;
				case V: strcpy(name, "V   ");
					break;
				case Vdag: strcpy(name, "Vdag");
					break;
				case U3: strcpy(name, "U3  ");
					break;
				case U2: strcpy(name, "U2  ");
					break;
				case U1: strcpy(name, "U1  ");
					break;
				case RX: strcpy(name, "RX  ");
					break;
				case RY: strcpy(name, "RY  ");
					break;
				case RZ: strcpy(name, "RZ  ");
					break;
				case SWAP: strcpy(name, "SWAP");
					break;
				case P: strcpy(name, "P   ");
					break;
				case Pdag: strcpy(name, "Pdag");
					break;
				default: std::cerr << "This constructor shall not be called for gate type (index) " << (int) gate << std::endl;
					exit(1);
			}
		}

		void checkUgate() {
			if (gate == U1) {
				gate = parseU1(parameter[0]);
			} else if (gate == U2) {
				gate = parseU2(parameter[0], parameter[1]);
			} else if (gate == U3) {
				gate = parseU3(parameter[0], parameter[1], parameter[2]);
			}
		}

	public:
		StandardOperation() = default;

		// Standard Constructors
		StandardOperation(unsigned short nq, unsigned short target, Gate g, fp lambda = 0, fp phi = 0, fp theta = 0) : gate(g) {
			nqubits = nq;
			line.fill(-1);
			line[target] = 2;
			parameter[0] = lambda;
			parameter[1] = phi;
			parameter[2] = theta;
			checkUgate();
			setName();
		}
		StandardOperation(unsigned short nq, const std::vector<unsigned short>& targets, Gate g, fp lambda = 0, fp phi = 0, fp theta = 0) : gate(g) {
			nqubits = nq;
			line.fill(-1);
			for (auto t: targets)
				line[t] = 2;
			if (targets.size() > 1)
				multiTarget = true;
			parameter[0] = lambda;
			parameter[1] = phi;
			parameter[2] = theta;
			checkUgate();
			setName();
		};

		StandardOperation(unsigned short nq, short control, unsigned short target, Gate g, fp lambda = 0, fp phi = 0, fp theta = 0) : gate(g) {
			nqubits = nq;
			line.fill(-1);
			line[std::abs(control)] = std::signbit(control) ? 0 : 1;
			controlled = true;
			line[target] = 2;
			parameter[0] = lambda;
			parameter[1] = phi;
			parameter[2] = theta;
			checkUgate();
			setName();
		}

		StandardOperation(unsigned short nq, short control, const std::vector<unsigned short>& targets, Gate g, fp lambda = 0, fp phi = 0, fp theta = 0) : gate(g) {
			nqubits = nq;
			line.fill(-1);
			line[std::abs(control)] = std::signbit(control) ? 0 : 1;
			controlled = true;
			for (auto t: targets)
				line[t] = 2;
			if (targets.size() > 1)
				multiTarget = true;
			parameter[0] = lambda;
			parameter[1] = phi;
			parameter[2] = theta;
			checkUgate();
			setName();
		};

		StandardOperation(unsigned short nq, const std::vector<short>& controls, unsigned short target, Gate g, fp lambda = 0, fp phi = 0, fp theta = 0) : gate(g) {
			nqubits = nq;
			line.fill(-1);
			for (auto c: controls)
				line[std::abs(c)] = std::signbit(c) ? 0 : 1;
			if (!controls.empty())
				controlled = true;
			line[target] = 2;
			parameter[0] = lambda;
			parameter[1] = phi;
			parameter[2] = theta;
			checkUgate();
			setName();
		}

		StandardOperation(unsigned short nq, const std::vector<short>& controls, const std::vector<unsigned short>& targets, Gate g, fp lambda = 0, fp phi = 0, fp theta = 0) : gate(g) {
			nqubits = nq;
			line.fill(-1);
			for (auto c: controls)
				line[std::abs(c)] = std::signbit(c) ? 0 : 1;
			if (!controls.empty())
				controlled = true;
			for (auto t: targets)
				line[t] = 2;
			if (targets.size() > 1)
				multiTarget = true;
			parameter[0] = lambda;
			parameter[1] = phi;
			parameter[2] = theta;
			checkUgate();
			setName();
		}

		// MCT Constructor
		StandardOperation(unsigned short nq, const std::vector<short>& controls, unsigned short target) : StandardOperation(nq, controls, target, X) {
			parameter[0] = 1;
			parameter[1] = controls.size();
			parameter[2] = target;
		}

		// MCF (cSWAP) and Peres Constructor
		StandardOperation(unsigned short nq, const std::vector<short>& controls, unsigned short target0, unsigned short target1, Gate g) : StandardOperation(nq, controls, { target0, target1 }, g) {
			parameter[0] = target0;
			parameter[1] = target1;
		}

		Gate getGate() const { return gate; }

		dd::Edge getDD(std::unique_ptr<dd::Package>& dd) override {
			dd::Edge e{ };
			GateMatrix gm;
			switch (gate) {
				case I:	gm = Imat; break;
				case H: gm = Hmat; break;
				case X:
					if (parameter[0] == 1) {
						e = dd->TTlookup(nqubits, parameter[1], parameter[2], line.data());
						if (e.p == nullptr) {
							e = dd->makeGateDD(Xmat, nqubits, line);
							dd->TTinsert(nqubits, parameter[1], parameter[2], line.data(), e);
						}
						return e;
					}
					gm = Xmat; break;
				case Y: gm = Ymat; break;
				case Z: gm = Zmat; break;
				case S: gm = Smat; break;
				case Sdag: gm = Sdagmat; break;
				case T: gm = Tmat; break;
				case Tdag: gm = Tdagmat; break;
				case V: gm = Vmat; break;
				case Vdag: gm = Vdagmat; break;
				case U3: gm = U3mat(parameter[0], parameter[1], parameter[2]); break;
				case U2: gm = U2mat(parameter[0], parameter[1]); break;
				case U1: gm = RZmat(parameter[0]); break;
				case RX: gm = RXmat(parameter[0]); break;
				case RY: gm = RYmat(parameter[0]); break;
				case RZ: gm = RZmat(parameter[0]); break;
				case SWAP: line[parameter[0]] = 1;
					e = dd->makeGateDD(Xmat, nqubits, line);
					line[parameter[0]] = 2;
					line[parameter[1]] = 1;
					e = dd->multiply(e, dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e));
					line[parameter[1]] = 2;
					return e;
				case P: line[parameter[1]] = 1;
					e = dd->makeGateDD(Xmat, nqubits, line);
					line[parameter[1]] = 2;
					line[parameter[0]] = -1;
					e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
					line[parameter[0]] = 2;
					return e;
				case Pdag: line[parameter[0]] = -1;
					e = dd->makeGateDD(Xmat, nqubits, line);
					line[parameter[0]] = 2;
					line[parameter[1]] = 1;
					e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
					line[parameter[1]] = 2;
					return e;
				default: std::cerr << "DD for gate" << name << " not available!" << std::endl;
					exit(1);
			}
			if (multiTarget && !controlled)
				return dd->makeMultiTargetSingleQubitDD(gm, nqubits, line);
			else
				return dd->makeGateDD(gm, nqubits, line);
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd) override {
			dd::Edge e{ };
			GateMatrix gm;
			switch (gate) {
				case I: gm = Imat; break;
				case H: gm = Hmat; break;
				case X:
					if (parameter[0] == 1) {
						e = dd->TTlookup(nqubits, parameter[1], parameter[2], line.data());
						if (e.p == nullptr) {
							e = dd->makeGateDD(Xmat, nqubits, line);
							dd->TTinsert(nqubits, parameter[1], parameter[2], line.data(), e);
						}
						return e;
					}
					gm = Xmat; break;
				case Y: gm = Ymat; break;
				case Z: gm = Zmat; break;
				case S: gm = Sdagmat; break;
				case Sdag: gm = Smat; break;
				case T: gm = Tdagmat; break;
				case Tdag: gm = Tmat; break;
				case V: gm = Vdagmat; break;
				case Vdag: gm = Vmat; break;
				case U3: gm = U3dagmat(parameter[0], parameter[1], parameter[2]); break;
				case U2: gm = U2dagmat(parameter[0], parameter[1]); break;
				case U1: gm = RZmat(-parameter[0]); break;
				case RX: gm = RXmat(-parameter[0]); break;
				case RY: gm = RYmat(-parameter[0]); break;
				case RZ: gm = RZmat(-parameter[0]); break;
				case SWAP: line[parameter[0]] = 1;
					e = dd->makeGateDD(Xmat, nqubits, line);
					line[parameter[0]] = 2;
					line[parameter[1]] = 1;
					e = dd->multiply(e, dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e));
					line[parameter[1]] = 2;
					return e;
				case P: line[parameter[0]] = -1;
					e = dd->makeGateDD(Xmat, nqubits, line);
					line[parameter[0]] = 2;
					line[parameter[1]] = 1;
					e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
					line[parameter[1]] = 2;
					return e;
				case Pdag: line[parameter[1]] = 1;
					e = dd->makeGateDD(Xmat, nqubits, line);
					line[parameter[1]] = 2;
					line[parameter[0]] = -1;
					e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
					line[parameter[0]] = 2;
					return e;
				default: std::cerr << "Inverse DD for gate " << name << " not available!" << std::endl;
					exit(1);
			}
			if (multiTarget && !controlled)
				return dd->makeMultiTargetSingleQubitDD(gm, nqubits, line);
			else
				return dd->makeGateDD(gm, nqubits, line);
		}
	};

}
#endif //INTERMEDIATEREPRESENTATION_STANDARDOPERATION_H
