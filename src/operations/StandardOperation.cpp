/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "operations/StandardOperation.hpp"

namespace qc {
    /***
     * Protected Methods
     ***/
    Gate StandardOperation::parseU3(fp& lambda, fp& phi, fp& theta) {
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

	Gate StandardOperation::parseU2(fp& lambda, fp& phi) {
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

	Gate StandardOperation::parseU1(fp& lambda) {
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

	void StandardOperation::setName() {
		switch (gate) {
			case I: 
                strcpy(name, "I   ");
				break;
			case H: 
                strcpy(name, "H   ");
				break;
			case X: 
                strcpy(name, "X   ");
				break;
			case Y: 
                strcpy(name, "Y   ");
				break;
			case Z: 
                strcpy(name, "Z   ");
				break;
			case S: 
                strcpy(name, "S   ");
				break;
			case Sdag: 
                strcpy(name, "Sdag");
				break;
			case T: 
                strcpy(name, "T   ");
				break;
			case Tdag: 
                strcpy(name, "Tdag");
				break;
			case V: 
                strcpy(name, "V   ");
				break;
			case Vdag: 
                strcpy(name, "Vdag");
				break;
			case U3: 
                strcpy(name, "U3  ");
				break;
			case U2: 
                strcpy(name, "U2  ");
				break;
			case U1: 
                strcpy(name, "U1  ");
				break;
			case RX: 
                strcpy(name, "RX  ");
				break;
			case RY: 
                strcpy(name, "RY  ");
				break;
			case RZ: 
                strcpy(name, "RZ  ");
				break;
			case SWAP: 
                strcpy(name, "SWAP");
				break;
			case iSWAP: 
                strcpy(name, "iSWP");
				break;
			case P: 
                strcpy(name, "P   ");
				break;
			case Pdag: 
                strcpy(name, "Pdag");
				break;
			default: 
                std::cerr << "This constructor shall not be called for gate type (index) " << (int) gate << std::endl;
				exit(1);
		}
	}

	void StandardOperation::checkUgate() {
		if (gate == U1) {
			gate = parseU1(parameter[0]);
		} else if (gate == U2) {
			gate = parseU2(parameter[0], parameter[1]);
		} else if (gate == U3) {
			gate = parseU3(parameter[0], parameter[1], parameter[2]);
		}
	}

	void StandardOperation::setup(unsigned short nq, fp par0, fp par1, fp par2) {
		nqubits = nq;
		parameter[0] = par0;
		parameter[1] = par1;
		parameter[2] = par2;
		checkUgate();
		setName();
	}

	dd::Edge StandardOperation::getSWAPDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const {
		dd::Edge e{ };

		line[permutation.at(targets[0])] = LINE_CONTROL_POS;
		e = dd->makeGateDD(Xmat, nqubits, line);

		line[permutation.at(targets[0])] = LINE_TARGET;
		line[permutation.at(targets[1])] = LINE_CONTROL_POS;
		e = dd->multiply(e, dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e));

		line[permutation.at(targets[1])] = LINE_TARGET;
		return e;
    }

	dd::Edge StandardOperation::getPDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const {
		dd::Edge e{ };

		line[permutation.at(targets[1])] = LINE_CONTROL_POS;
		e = dd->makeGateDD(Xmat, nqubits, line);

		line[permutation.at(targets[0])] = LINE_DEFAULT;
		line[permutation.at(targets[1])] = LINE_TARGET;
		e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);

		line[permutation.at(targets[0])] = LINE_TARGET;
		return e;
	}

	dd::Edge StandardOperation::getPdagDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const {
		dd::Edge e{ };

		line[permutation.at(targets[0])] = LINE_DEFAULT;
		e = dd->makeGateDD(Xmat, nqubits, line);

		line[permutation.at(targets[0])] = LINE_TARGET;
		line[permutation.at(targets[1])] = LINE_CONTROL_POS;
		e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);

		line[permutation.at(targets[1])] = LINE_TARGET;
		return e;
    }

	dd::Edge StandardOperation::getiSWAPDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const {
		// TODO: this can be simplified since H-CX-H == CZ

    	dd::Edge e{ };

		e = getSWAPDD(dd, line, permutation);


		line[permutation.at(targets[1])] = LINE_DEFAULT;
		e = dd->multiply(e, dd->makeGateDD(Smat, nqubits, line));

		line[permutation.at(targets[0])] = LINE_DEFAULT;
		line[permutation.at(targets[1])] = LINE_TARGET;
		e = dd->multiply(e, dd->makeGateDD(Smat, nqubits, line));
		e = dd->multiply(e, dd->makeGateDD(Hmat, nqubits, line));

		line[permutation.at(targets[0])] = LINE_CONTROL_POS;
		e = dd->multiply(e, dd->makeGateDD(Xmat, nqubits, line));

		line[permutation.at(targets[0])] = LINE_DEFAULT;
		e = dd->multiply(e, dd->makeGateDD(Hmat, nqubits, line));

		line[permutation.at(targets[0])] = LINE_TARGET;
		return e;
    }

	dd::Edge StandardOperation::getiSWAPinvDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const {
		// TODO: this can be simplified since H-CX-H == CZ

		dd::Edge e{ };

		line[permutation.at(targets[0])] = LINE_DEFAULT;
		e = dd->makeGateDD(Hmat, nqubits, line);

		line[permutation.at(targets[0])] = LINE_CONTROL_POS;
		e = dd->multiply(e, dd->makeGateDD(Xmat, nqubits, line));

		line[permutation.at(targets[0])] = LINE_DEFAULT;
		e = dd->multiply(e, dd->makeGateDD(Hmat, nqubits, line));
		e = dd->multiply(e, dd->makeGateDD(Sdagmat, nqubits, line));

		line[permutation.at(targets[0])] = LINE_TARGET;
		line[permutation.at(targets[1])] = LINE_DEFAULT;
		e = dd->multiply(e, dd->makeGateDD(Sdagmat, nqubits, line));

		line[permutation.at(targets[1])] = LINE_TARGET;
		e = dd->multiply(e, getSWAPDD(dd, line, permutation));

		return e;
    }

	dd::Edge StandardOperation::getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, bool inverse, const std::map<unsigned short, unsigned short>& permutation
	) const {
		dd::Edge e{ };
		GateMatrix gm;
		//TODO add assertions ?
		switch (gate) {
			case I:	gm = Imat; break;
			case H: gm = Hmat; break;
			case X:
				if (controls.size() > 1) { //Toffoli //TODO > 0 (include CNOT?)
					e = dd->TTlookup(nqubits, static_cast<unsigned short>(controls.size()), targets[0], line.data());
					if (e.p == nullptr) {
						e = dd->makeGateDD(Xmat, nqubits, line);
						dd->TTinsert(nqubits, static_cast<unsigned short>(controls.size()), targets[0], line.data(), e);
					}
					return e;
				}
				gm = Xmat;
				break;
			case Y:    gm = Ymat; break;
			case Z:    gm = Zmat; break;
			case S:    gm = inverse? Sdagmat: Smat; break;
			case Sdag: gm = inverse? Smat: Sdagmat; break;
			case T:    gm = inverse? Tdagmat: Tmat; break;
			case Tdag: gm = inverse? Tmat: Tdagmat; break;
			case V:    gm = inverse? Vdagmat: Vmat; break;
			case Vdag: gm = inverse? Vmat: Vdagmat; break;
			case U3:   gm = inverse? U3mat(-parameter[1], -parameter[0], -parameter[2]): U3mat(parameter[0], parameter[1], parameter[2]); break;
			case U2:   gm = inverse? U2mat(-parameter[1]+PI, -parameter[0]-PI): U2mat(parameter[0], parameter[1]); break;
			case U1:   gm = inverse? RZmat(-parameter[0]): RZmat(parameter[0]); break;
			case RX:   gm = inverse? RXmat(-parameter[0]): RXmat(parameter[0]); break;
			case RY:   gm = inverse? RYmat(-parameter[0]): RYmat(parameter[0]); break;
			case RZ:   gm = inverse? RZmat(-parameter[0]): RZmat(parameter[0]); break;
			case SWAP:
				return getSWAPDD(dd, line, permutation);
			case iSWAP:
				if(inverse) {
					return getiSWAPinvDD(dd, line, permutation);
				} else {
					return getiSWAPDD(dd, line, permutation);
				}
			case P:
				if (inverse) {
					return getPdagDD(dd, line, permutation);
				} else {
					return getPDD(dd, line, permutation);
				}
			case Pdag:
				if (inverse) {
					return getPDD(dd, line, permutation);
				} else {
					return getPdagDD(dd, line, permutation);
				}
			default:
				std::cerr << "DD for gate" << name << " not available!" << std::endl;
				exit(1);
		}
		if (multiTarget && !controlled) {
			std::cerr << "Multi target gates not implemented yet!" << std::endl;
			exit(1);
		} else {
			return dd->makeGateDD(gm, nqubits, line);
		}
    }

    /***
     * Constructors
     ***/
	StandardOperation::StandardOperation(unsigned short nq, unsigned short target, Gate g, fp lambda, fp phi, fp theta) 
		: gate(g) {
		
		setup(nq, lambda, phi, theta);
		targets.push_back(target);
	}

	StandardOperation::StandardOperation(unsigned short nq, const std::vector<unsigned short>& targets, Gate g, fp lambda, fp phi, fp theta) 
		: gate(g) {
		
		setup(nq, lambda, phi, theta);
		this->targets = targets;
		if (targets.size() > 1)
			multiTarget = true;
	}

	StandardOperation::StandardOperation(unsigned short nq, Control control, unsigned short target, Gate g, fp lambda, fp phi, fp theta) 
		: StandardOperation(nq, target, g, lambda, phi, theta) {
		
		//line[control.qubit] = control.type == Control::pos? LINE_CONTROL_POS: LINE_CONTROL_NEG;
		controlled   = true;
		controls.push_back(control);
	}

	StandardOperation::StandardOperation(unsigned short nq, Control control, const std::vector<unsigned short>& targets, Gate g, fp lambda, fp phi, fp theta) 
		: StandardOperation(nq, targets, g, lambda, phi, theta) {
		
		//line[control.qubit] = control.type == Control::pos? LINE_CONTROL_POS: LINE_CONTROL_NEG;
		controlled = true;
		controls.push_back(control);
	}

	StandardOperation::StandardOperation(unsigned short nq, const std::vector<Control>& controls, unsigned short target, Gate g, fp lambda, fp phi, fp theta) 
		: StandardOperation(nq, target, g, lambda, phi, theta) {
		
		this->controls = controls;
		if (!controls.empty())
			controlled = true;
	}

	StandardOperation::StandardOperation(unsigned short nq, const std::vector<Control>& controls, const std::vector<unsigned short>& targets, Gate g, fp lambda, fp phi, fp theta) 
		: StandardOperation(nq, targets, g, lambda, phi, theta) {
		
		this->controls = controls;
		if (!controls.empty())
			controlled = true;
	}

	// MCT Constructor
	StandardOperation::StandardOperation(unsigned short nq, const std::vector<Control>& controls, unsigned short target) 
		: StandardOperation(nq, controls, target, X) {
	}

	// MCF (cSWAP) and Peres Constructor
	StandardOperation::StandardOperation(unsigned short nq, const std::vector<Control>& controls, unsigned short target0, unsigned short target1, Gate g) 
		: StandardOperation(nq, controls, { target0, target1 }, g) {
	}

	/***
     * Public Methods
    ***/	
	void StandardOperation::dumpOpenQASM(std::ofstream& of, const regnames_t& qreg, const regnames_t&) const {
		std::ostringstream op;
		op << std::setprecision(std::numeric_limits<fp>::digits10);
		if((controls.size() > 1 && gate != X) || controls.size() > 2) {
			std::cout << "[WARNING] Multiple controlled gates are not natively suppported by OpenQASM. "
			<< "However, this library can parse .qasm files with multiple controlled gates (e.g., cccx) correctly. "
			<< "Thus, while not valid vanilla OpenQASM, the dumped file will work with this library. " << std::endl;
		}

		for (const auto& c: controls) {
			of << "c";
		}

		switch (gate) {
			case I: 
               	op << "id";
				break;
			case H:
				of << "h";
				break;
			case X:
				of << "x";
				break;
			case Y:
				of << "y";
				break;
			case Z:
				op << "z";
				break;
			case S:
				if(!controls.empty()) {
					op << "u1(pi/2)";
				} else {
					op << "s";
				}
				break;
			case Sdag:
				if(!controls.empty()) {
					op << "u1(-pi/2)";
				} else {
					op << "sdg";
				}
				break;
			case T:
				if(!controls.empty()) {
					op << "u1(pi/4)";
				} else {
					op << "t";
				}
				break;
			case Tdag:
				if(!controls.empty()) {
					op << "u1(-pi/4)";
				} else {
					op << "tdg";
				}
				break;
			case V:
				op << "u3(pi/2, -pi/2, pi/2)";
				break;
			case Vdag:
				op << "u3(pi/2, pi/2, -pi/2)";
				break;
			case U3: 
				op << "u3(" << parameter[2] << "," << parameter[1] << "," << parameter[0] << ")";
				break;
			case U2:
				op << "u2(" << parameter[1] << "," << parameter[0] << ")";
				break;
			case U1: 
				op << "u1(" << parameter[0] << ")";
				break;
			case RX:
				op << "rx(" << parameter[0] << ")";
				break;
			case RY:
				op << "ry(" << parameter[0] << ")";
				break;
			case RZ: 
				op << "rz(" << parameter[0] << ")";
				break;
			case SWAP:
				of << "swap";
				for (const auto& c: controls)
					of << " " << qreg[c.qubit].second << ",";
				of << " " << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ";" << std::endl;
				return;
			case iSWAP:
				of << "swap";
				for (const auto& c: controls)
					of << " " << qreg[c.qubit].second << ",";
				of << " " << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ";" << std::endl;

				for (const auto& c: controls)
					of << "c";
				of << "s";
				for (const auto& c: controls)
					of << " " << qreg[c.qubit].second << ",";
				of << " " << qreg[targets[0]].second << ";"  << std::endl;


				for (const auto& c: controls)
					of << "c";
				of << "s";
				for (const auto& c: controls)
					of << " " << qreg[c.qubit].second << ",";
				of << " " << qreg[targets[1]].second << ";"  << std::endl;

				for (const auto& c: controls)
					of << "c";
                of << "cz";
				for (const auto& c: controls)
					of << " " << qreg[c.qubit].second << ",";
                of << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ";" << std::endl;
				return;
			case P: 
                of << "ccx";
				for (const auto& c: controls)
					of << " " << qreg[c.qubit].second << ",";
                of << " " << qreg[controls[0].qubit].second << ", " << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ";" << std::endl;

				for (const auto& c: controls)
					of << "c";
                of << "cx";
				for (const auto& c: controls)
					of << " " << qreg[c.qubit].second << ",";
                of << " " << qreg[controls[0].qubit].second << ", " << qreg[targets[1]].second << ";" << std::endl;
				return;
			case Pdag:
				of << "cx";
				for (const auto& c: controls)
					of << " " << qreg[c.qubit].second << ",";
				of << " " << qreg[controls[0].qubit].second << ", " << qreg[targets[1]].second << ";" << std::endl;

				for (const auto& c: controls)
					of << "c";
				of << "ccx";
				for (const auto& c: controls)
					of << " " << qreg[c.qubit].second << ",";
				of << " " << qreg[controls[0].qubit].second << ", " << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ";" << std::endl;
				return;
			default: 
                std::cerr << "gate type (index) " << (int) gate << " could not be converted to OpenQASM" << std::endl;
		}
		of << op.str();
		for (const auto& c: controls) {
			of << " " << qreg[c.qubit].second << ",";
		}
        for(auto target: targets) {
			of << " " << qreg[target].second << ";" << std::endl;
		}
	}

	void StandardOperation::dumpReal([[maybe_unused]] std::ofstream& of) const {}

	void StandardOperation::dumpQiskit(std::ofstream& of, const regnames_t& qreg,[[maybe_unused]] const regnames_t& creg, const char* anc_reg_name) const {
		std::ostringstream op;
		if (targets.size() > 2 || (targets.size() > 1 && gate != SWAP && gate != iSWAP && gate != P && gate != Pdag)) {
			std::cerr << "Multiple targets are not supported in general at the moment" << std::endl;
		}
		switch (gate) {
			case I:
				op << "qc.iden(";
				break;
			case H:
				switch(controls.size()) {
					case 0:
						op << "qc.h(";
						break;
					case 1:
						op << "qc.ch(" << qreg[controls[0].qubit].second << ", ";
						break;
					default:
						std::cerr << "Multi-controlled H gate currently not supported" << std::endl;
				}
				break;
			case X:
				switch(controls.size()) {
					case 0:
						op << "qc.x(";
						break;
					case 1:
						op << "qc.cx(" << qreg[controls[0].qubit].second << ", ";
						break;
					case 2:
						op << "qc.ccx(" << qreg[controls[0].qubit].second << ", " << qreg[controls[1].qubit].second << ", ";
						break;
					default:
						op << "qc.mct([";
						for (const auto& control:controls) {
							op << qreg[control.qubit].second << ", ";
						}
						op << "], " << qreg[targets[0]].second << ", " << anc_reg_name << ", mode='basic')" << std::endl;
						of << op.str();
						return;
				}
				break;
			case Y:
				switch(controls.size()) {
					case 0:
						op << "qc.y(";
						break;
					case 1:
						op << "qc.cy(" << qreg[controls[0].qubit].second << ", ";
						break;
					default:
						std::cerr << "Multi-controlled Y gate currently not supported" << std::endl;
				}
				break;
			case Z:
				if (!controls.empty()) {
					op << "qc.mcu1(pi, [";
					for (const auto& control:controls) {
						op << qreg[control.qubit].second << ", ";
					}
					op << "], ";
				} else {
					op << "qc.z(";
				}
				break;
			case S:
				if (!controls.empty()) {
					op << "qc.mcu1(pi/2, [";
					for (const auto& control:controls) {
						op << qreg[control.qubit].second << ", ";
					}
					op << "], ";
				} else {
					op << "qc.s(";
				}
				break;
			case Sdag:
				if (!controls.empty()) {
					op << "qc.mcu1(-pi/2, [";
					for (const auto& control:controls) {
						op << qreg[control.qubit].second << ", ";
					}
					op << "], ";
				} else {
					op << "qc.sdg(";
				}
				break;
			case T:
				if (!controls.empty()) {
					op << "qc.mcu1(pi/4, [";
					for (const auto& control:controls) {
						op << qreg[control.qubit].second << ", ";
					}
					op << "], ";
				} else {
					op << "qc.t(";
				}
				break;
			case Tdag:
				if (!controls.empty()) {
					op << "qc.mcu1(-pi/4, [";
					for (const auto& control:controls) {
						op << qreg[control.qubit].second << ", ";
					}
					op << "], ";
				} else {
					op << "qc.tdg(";
				}
				break;
			case V:
				switch(controls.size()) {
					case 0:
						op << "qc.u3(pi/2, -pi/2, pi/2, ";
						break;
					case 1:
						op << "qc.cu3(pi/2, -pi/2, pi/2, " << qreg[controls[0].qubit].second << ", ";
						break;
					default:
						std::cerr << "Multi-controlled V gate currently not supported" << std::endl;
				}
				break;
			case Vdag:
				switch(controls.size()) {
					case 0:
						op << "qc.u3(pi/2, pi/2, -pi/2, ";
						break;
					case 1:
						op << "qc.cu3(pi/2, pi/2, -pi/2, " << qreg[controls[0].qubit].second << ", ";
						break;
					default:
						std::cerr << "Multi-controlled Vdag gate currently not supported" << std::endl;
				}
				break;
			case U3:
				switch(controls.size()) {
					case 0:
						op << "qc.u3(" << parameter[2] << ", " << parameter[1] << ", " << parameter[0] << ", ";
						break;
					case 1:
						op << "qc.cu3(" << parameter[2] << ", " << parameter[1] << ", " << parameter[0] << ", " << qreg[controls[0].qubit].second << ", ";
						break;
					default:
						std::cerr << "Multi-controlled U3 gate currently not supported" << std::endl;
				}
				break;
			case U2:
				switch(controls.size()) {
					case 0:
						op << "qc.u3(pi/2, " << parameter[1] << ", " << parameter[0] << ", ";
						break;
					case 1:
						op << "qc.cu3(pi/2, " << parameter[1] << ", " << parameter[0] << ", " << qreg[controls[0].qubit].second << ", ";
						break;
					default:
						std::cerr << "Multi-controlled U2 gate currently not supported" << std::endl;
				}
				break;
			case U1:
				if (!controls.empty()) {
					op << "qc.mcu1(" << parameter[0] << ", [";
					for (const auto& control:controls) {
						op << qreg[control.qubit].second << ", ";
					}
					op << "], ";
				} else {
					op << "qc.u1(" << parameter[0] << ", ";
				}
				break;
			case RX:
				if (!controls.empty()) {
					op << "qc.mcrx(" << parameter[0] << ", [";
					for (const auto& control:controls) {
						op << qreg[control.qubit].second << ", ";
					}
					op << "], ";
				} else {
					op << "qc.rx(" << parameter[0] << ", ";
				}
				break;
			case RY:
				if (!controls.empty()) {
					op << "qc.mcry(" << parameter[0] << ", [";
					for (const auto& control:controls) {
						op << qreg[control.qubit].second << ", ";
					}
					op << "], ";
				} else {
					op << "qc.ry(" << parameter[0] << ", ";
				}
				break;
			case RZ:
				if (!controls.empty()) {
					op << "qc.mcrz(" << parameter[0] << ", [";
					for (const auto& control:controls) {
						op << qreg[control.qubit].second << ", ";
					}
					op << "], ";
				} else {
					op << "qc.rz(" << parameter[0] << ", ";
				}
				break;
			case SWAP:
				switch(controls.size()) {
					case 0:
						of << "qc.swap(" << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ")" << std::endl;
						break;
					case 1:
						of << "qc.cswap(" << qreg[controls[0].qubit].second << ", " << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ")" << std::endl;
						break;
					default:
						of << "qc.cx(" << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ")" << std::endl;
						of << "qc.mct([";
						for (const auto& control:controls) {
							of << qreg[control.qubit].second << ", ";
						}
						of << qreg[targets[0]].second << "], " << qreg[targets[1]].second << ", " << anc_reg_name << ", mode='basic')" << std::endl;
						of << "qc.cx(" << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ")" << std::endl;
						break;
				}
				return;
			case iSWAP:
				switch(controls.size()) {
					case 0:
						of << "qc.swap(" << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ")" << std::endl;
						of << "qc.s(" << qreg[targets[0]].second << ")" << std::endl;
						of << "qc.s(" << qreg[targets[1]].second << ")" << std::endl;
						of << "qc.cz(" << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ")" << std::endl;
						break;
					case 1:
						of << "qc.cswap(" << qreg[controls[0].qubit].second << ", " << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ")" << std::endl;
						of << "qc.cu1(pi/2, " << qreg[controls[0].qubit].second << ", " << qreg[targets[0]].second << ")" << std::endl;
						of << "qc.cu1(pi/2, " << qreg[controls[0].qubit].second << ", " << qreg[targets[1]].second << ")" << std::endl;
						of << "qc.mcu1(pi, [" << qreg[controls[0].qubit].second << ", " << qreg[targets[0]].second << "], " << qreg[targets[1]].second << ")" << std::endl;
						break;
					default:
						std::cerr << "Multi-controlled iSWAP gate currently not supported" << std::endl;
				}
				return;
			case P:
				of << "qc.ccx(" << qreg[controls[0].qubit].second << ", " << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ")" << std::endl;
				of << "qc.cx(" << qreg[controls[0].qubit].second << ", " << qreg[targets[1]].second << ")" << std::endl;
				return;
			case Pdag:
				of << "qc.cx(" << qreg[controls[0].qubit].second << ", " << qreg[targets[1]].second << ")" << std::endl;
				of << "qc.ccx(" << qreg[controls[0].qubit].second << ", " << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ")" << std::endl;
				return;
			default:
				std::cerr << "gate type (index) " << (int) gate << " could not be converted to qiskit" << std::endl;
		}
		of << op.str() << qreg[targets[0]].second << ")" << std::endl;
	}


	dd::Edge StandardOperation::getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const {
		setLine(line);
		dd::Edge e = getDD(dd, line, false);
		resetLine(line);
		return e;
	}

	dd::Edge StandardOperation::getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const {

		if(gate == SWAP && controls.empty()) {
			auto target0 = targets.at(0);
			auto target1 = targets.at(1);
			// update permutation
			auto tmp = permutation.at(target0);
			permutation.at(target0) = permutation.at(target1);
			permutation.at(target1) = tmp;
			return dd->makeIdent(0, short(nqubits-1));
		}

		setLine(line, permutation);
		dd::Edge e = getDD(dd, line, false, permutation);
		resetLine(line, permutation);
		return e;
	}

	dd::Edge StandardOperation::getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const {
		setLine(line);
		dd::Edge e = getDD(dd, line, true);
		resetLine(line);
		return e;
	}

	dd::Edge StandardOperation::getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const {

		if(gate == SWAP && controls.empty()) {
			auto target0 = targets.at(0);
			auto target1 = targets.at(1);
			// update permutation
			auto tmp = permutation.at(target0);
			permutation.at(target0) = permutation.at(target1);
			permutation.at(target1) = tmp;
			return dd->makeIdent(0, short(nqubits-1));
		}

		setLine(line, permutation);
		dd::Edge e = getDD(dd, line, true, permutation);
		resetLine(line, permutation);
		return e;
	}
}
