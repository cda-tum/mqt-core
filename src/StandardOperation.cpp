//
// Created by Lukas Burgholzer on 30.09.19.
//

#include "StandardOperation.h"

StandardOperation::StandardOperation(short nq, const std::vector<short>& controls, const std::vector<short>& targets, Gate g, double lambda, double phi, double theta) : gate(g) {
	nqubits = nq;
	for (int i = 0; i < nqubits; ++i) {
		line[i] = -1;
	}
	for (const auto& control:controls)
		line[std::abs(control)] = ((control >= 0) ? 1 : 0);

	for (const auto& target:targets)
		line[target] = 2;

	parameter[0] = lambda;
	parameter[1] = phi;
	parameter[2] = theta;

	switch (gate) {
		case I: strcpy(name, "I");
			break;
		case H: strcpy(name, "H");
			break;
		case X: strcpy(name, "X");
			break;
		case Y: strcpy(name, "Y");
			break;
		case Z: strcpy(name, "Z");
			break;
		case S: strcpy(name, "S");
			break;
		case Sdag: strcpy(name, "Sdag");
			break;
		case T: strcpy(name, "T");
			break;
		case Tdag: strcpy(name, "Tdag");
			break;
		case V: strcpy(name, "V");
			break;
		case Vdag: strcpy(name, "Vdag");
			break;
		case U3: strcpy(name, "U3");
			break;
		case U2: strcpy(name, "U2");
			break;
		case U1: strcpy(name, "U1");
			break;
		case RX: strcpy(name, "RX");
			break;
		case RY: strcpy(name, "RY");
			break;
		case RZ: strcpy(name, "RZ");
			break;
		case SWAP: strcpy(name, "SWAP");
			break;
		case P: strcpy(name, "P");
			break;
		case Pdag: strcpy(name, "Pdag");
			break;
		default:
			std::cerr << "This constructor shall not be called for gate type (index) " << (int)g << std::endl;
			exit(1);
	}
}

dd::Edge StandardOperation::getDD(dd::Package *dd) {
	dd::Edge e{ };
	switch (gate) {
		case I: return dd->makeGateDD(Imat, nqubits, line);
		case H: return dd->makeGateDD(Hmat, nqubits, line);
		case X:
			if (parameter[0] == 1){
				e = dd->TTlookup(nqubits, parameter[1], parameter[2], line.data());
				if (e.p == nullptr) {
					e = dd->makeGateDD(Xmat, nqubits, line);
					dd->TTinsert(nqubits, parameter[1], parameter[2], line.data(), e);
				}
				return e;
			}
			return dd->makeGateDD(Xmat, nqubits, line);
			
		case Y: return dd->makeGateDD(Ymat, nqubits, line); 
		case Z: return dd->makeGateDD(Zmat, nqubits, line); 
		case S: return dd->makeGateDD(Smat, nqubits, line); 
		case Sdag: return dd->makeGateDD(Sdagmat, nqubits, line); 
		case T: return dd->makeGateDD(Tmat, nqubits, line); 
		case Tdag: return dd->makeGateDD(Tdagmat, nqubits, line); 
		case V: return dd->makeGateDD(Vmat, nqubits, line); 
		case Vdag: return dd->makeGateDD(Vdagmat, nqubits, line); 
		case U3: return dd->makeGateDD(U3mat(parameter[0], parameter[1], parameter[2]), nqubits, line); 
		case U2: return dd->makeGateDD(U2mat(parameter[0], parameter[1]), nqubits, line); 
		case U1: return dd->makeGateDD(RZmat(parameter[0]), nqubits, line); 
		case RX: return dd->makeGateDD(RXmat(parameter[0]), nqubits, line); 
		case RY: return dd->makeGateDD(RYmat(parameter[0]), nqubits, line); 
		case RZ: return dd->makeGateDD(RZmat(parameter[0]), nqubits, line); 
		case SWAP:
			line[parameter[0]] = 1;
			e = dd->makeGateDD(Xmat, nqubits, line);
			line[parameter[0]] = 2;
			line[parameter[1]] = 1;
			e = dd->multiply(e, dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e));
			line[parameter[1]] = 2;
			return e;
		case P:
			line[parameter[1]] = 1;
			e = dd->makeGateDD(Xmat, nqubits, line);
			line[parameter[1]] = 2;
			line[parameter[0]] = -1;
			e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
			line[parameter[0]] = 2;
			return e;
		case Pdag:
			line[parameter[0]] = -1;
			e = dd->makeGateDD(Xmat, nqubits, line);
			line[parameter[0]] = 2;
			line[parameter[1]] = 1;
			e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
			line[parameter[1]] = 2;
			return e;
		case Measure:
			std::cerr << "No DD for measurement available!" << std::endl;
			exit(1);
		default:
			std::cerr << "DD for gate" << name << " not available!" << std::endl;
			exit(1);
	}
}

dd::Edge StandardOperation::getInverseDD(dd::Package *dd) {
	dd::Edge e{ };
	switch (gate) {
		case I:     return dd->makeGateDD(Imat, nqubits, line);
		case H:     return dd->makeGateDD(Hmat, nqubits, line);
		case X:
			if (parameter[0] == 1) {
				e = dd->TTlookup(nqubits, parameter[1], parameter[2], line.data());
				if (e.p == nullptr) {
					e = dd->makeGateDD(Xmat, nqubits, line);
					dd->TTinsert(nqubits, parameter[1], parameter[2], line.data(), e);
				}
				return e;
			}
			return dd->makeGateDD(Xmat, nqubits, line);
		case Y:     return dd->makeGateDD(Ymat, nqubits, line);
		case Z:     return dd->makeGateDD(Zmat, nqubits, line);
		case S:     return dd->makeGateDD(Sdagmat, nqubits, line);
		case Sdag:  return dd->makeGateDD(Smat, nqubits, line);
		case T:     return dd->makeGateDD(Tdagmat, nqubits, line);
		case Tdag:  return dd->makeGateDD(Tmat, nqubits, line);
		case V:     return dd->makeGateDD(Vdagmat, nqubits, line);
		case Vdag:  return dd->makeGateDD(Vmat, nqubits, line);
		case U3:	return dd->makeGateDD(U3dagmat(parameter[0], parameter[1], parameter[2]), nqubits, line);
		case U2:    return dd->makeGateDD(U2dagmat(parameter[0], parameter[1]), nqubits, line);
		case U1:	return dd->makeGateDD(RZmat(-parameter[0]), nqubits, line);
		case RX:	return dd->makeGateDD(RXmat(-parameter[0]), nqubits, line);
		case RY:	return dd->makeGateDD(RYmat(-parameter[0]), nqubits, line);
		case RZ:    return dd->makeGateDD(RZmat(-parameter[0]), nqubits, line);
		case SWAP:
			line[parameter[0]] = 1;
			e = dd->makeGateDD(Xmat, nqubits, line);
			line[parameter[0]] = 2;
			line[parameter[1]] = 1;
			e = dd->multiply(e, dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e));
			line[parameter[1]] = 2;
			return e;
		case P:
			line[parameter[0]] = -1;
			e = dd->makeGateDD(Xmat, nqubits, line);
			line[parameter[0]] = 2;
			line[parameter[1]] = 1;
			e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
			line[parameter[1]] = 2;
			return e;
		case Pdag:
			line[parameter[1]] = 1;
			e = dd->makeGateDD(Xmat, nqubits, line);
			line[parameter[1]] = 2;
			line[parameter[0]] = -1;
			e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
			line[parameter[0]] = 2;
			return e;
		case Measure:
			std::cerr << "No DD for measurement available!" << std::endl;
			exit(1);
		default:
			std::cerr << "Inverse DD for gate " << name << " not available!" << std::endl;
			exit(1);
	}
}
