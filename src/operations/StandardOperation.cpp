//
// Created by Lukas Burgholzer on 09.12.19.
//

#include "StandardOperation.hpp"

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
                strcpy(name, "iSWAP");
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

	dd::Edge StandardOperation::getSWAPDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) {
		dd::Edge e{ };
		line[targets[0]] = LINE_CONTROL_POS;
		e = dd->makeGateDD(Xmat, nqubits, line);
		line[targets[0]] = LINE_TARGET;
		line[targets[1]] = LINE_CONTROL_POS;
		e = dd->multiply(e, dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e));
		line[targets[1]] = LINE_TARGET;
		return e;
	}

	dd::Edge StandardOperation::getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, bool inverse) {
		dd::Edge e{ };
		GateMatrix gm;
		//TODO add assertions ? 
		switch (gate) {
			case I:	gm = Imat; break;
			case H: gm = Hmat; break;
			case X: 
				if (controls.size() > 1) { //Toffoli //TODO > 0 (include CNOT?)
					e = dd->TTlookup(nqubits, controls.size(), targets[0], line.data());
					if (e.p == nullptr) {
						e = dd->makeGateDD(Xmat, nqubits, line);
						dd->TTinsert(nqubits, controls.size(), targets[0], line.data(), e);
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
			case U3:   gm = inverse? U3dagmat(parameter[0], parameter[1], parameter[2]): U3mat(parameter[0], parameter[1], parameter[2]); break;
			case U2:   gm = inverse? U2dagmat(parameter[0], parameter[1]): U2mat(parameter[0], parameter[1]); break;
			case U1:   gm = inverse? RZmat(-parameter[0]): RZmat(parameter[0]); break;
			case RX:   gm = inverse? RXmat(-parameter[0]): RXmat(parameter[0]); break;
			case RY:   gm = inverse? RYmat(-parameter[0]): RYmat(parameter[0]); break;
			case RZ:   gm = inverse? RZmat(-parameter[0]): RZmat(parameter[0]); break;
			case SWAP: 
				return getSWAPDD(dd, line);
			case iSWAP: 
				if(inverse) {
					line[targets[0]] = LINE_DEFAULT;
					e = dd->multiply(e, dd->makeGateDD(Hmat, nqubits, line));
					line[targets[0]] = LINE_CONTROL_POS;
					e = dd->multiply(e, dd->makeGateDD(Xmat, nqubits, line));
					line[targets[0]] = LINE_DEFAULT;
					e = dd->multiply(e, dd->makeGateDD(Smat, nqubits, line));
					e = dd->multiply(e, dd->makeGateDD(Hmat, nqubits, line));
					line[targets[1]] = LINE_DEFAULT;
					e = dd->multiply(e, dd->makeGateDD(Smat, nqubits, line));
					e = dd->multiply(e, getSWAPDD(dd, line));
					line[targets[0]] = LINE_TARGET;
					return e;
				}

				e = getSWAPDD(dd, line);
				line[targets[1]] = LINE_DEFAULT;
				e = dd->multiply(e, dd->makeGateDD(Smat, nqubits, line));
				line[targets[0]] = LINE_DEFAULT;
				line[targets[1]] = LINE_TARGET;
				e = dd->multiply(e, dd->makeGateDD(Smat, nqubits, line));
				e = dd->multiply(e, dd->makeGateDD(Hmat, nqubits, line));
				line[targets[0]] = LINE_CONTROL_POS;
				e = dd->multiply(e, dd->makeGateDD(Xmat, nqubits, line));
				line[targets[0]] = LINE_DEFAULT;
				e = dd->multiply(e, dd->makeGateDD(Hmat, nqubits, line));
				
				line[targets[0]] = LINE_TARGET;
				return e;
			case P:
				if (inverse) {
					line[targets[0]] = LINE_DEFAULT;
					e = dd->makeGateDD(Xmat, nqubits, line);
					line[targets[0]] = LINE_TARGET;
					line[targets[1]] = LINE_CONTROL_POS;
					e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
					line[targets[1]] = LINE_TARGET;
					return e;
				}

				line[targets[1]] = LINE_CONTROL_POS;
				e = dd->makeGateDD(Xmat, nqubits, line);
				line[targets[1]] = LINE_TARGET;
				line[targets[0]] = LINE_DEFAULT;
				e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
				line[targets[0]] = LINE_TARGET;
				return e;
			case Pdag:
				if (inverse) {
					line[targets[1]] = LINE_CONTROL_POS;
					e = dd->makeGateDD(Xmat, nqubits, line);
					line[targets[1]] = LINE_TARGET;
					line[targets[0]] = LINE_DEFAULT;
					e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
					line[targets[0]] = LINE_TARGET;
					return e;
				}

				line[targets[0]] = LINE_DEFAULT;
				e = dd->makeGateDD(Xmat, nqubits, line);
				line[targets[0]] = LINE_TARGET;
				line[targets[1]] = LINE_CONTROL_POS;
				e = dd->multiply(dd->makeGateDD(Xmat, nqubits, line), e);
				line[targets[1]] = LINE_TARGET;
				return e;
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
		
		//parameter[0] = 1;
		//parameter[1] = controls.size();
		//parameter[2] = target;
	}

	// MCF (cSWAP) and Peres Constructor
	StandardOperation::StandardOperation(unsigned short nq, const std::vector<Control>& controls, unsigned short target0, unsigned short target1, Gate g) 
		: StandardOperation(nq, controls, { target0, target1 }, g) {
		
		//parameter[0] = target0;
		//parameter[1] = target1;
	}

	/***
     * Public Methods
    ***/	
	void StandardOperation::dumpOpenQASM(std::ofstream& of, const std::vector<std::string>& qreg, const std::vector<std::string>& creg) const {
		//TODO handle multiple controls
		std::ostringstream name;
		switch (gate) {
			case I: 
               	name << "id";
				break;
			case H: 
                name << "h";
				break;
			case X: 
				switch(controls.size()) {
					case 0:
                		name << "x";
						break;
					case 1:
               			name << "cx " << qreg[controls[0].qubit] << ", ";
						break;
					case 2:
               			name << "ccx " << qreg[controls[0].qubit] << ", " << qreg[controls[1].qubit] << ",";
						break;
					default:
						std::cerr << "MCT not yet supported" << std::endl;
				}
				break;
			case Y:
                name << "y";
				break;
			case Z: 
                name << "z";
				break;
			case S: 
                name << "s";
				break;
			case Sdag: 
                name << "sdg";
				break;
			case T: 
                name << "t";
				break;
			case Tdag: 
                name << "tdg";
				break;
			case V: 
				name << "u3(pi/2, -pi/2, pi/2)";
				break;
			case Vdag: 
				name << "u3(pi/2, pi/2, -pi/2)";				
				break;
			case U3: 
                name << "u3(" << parameter[2] << "," << parameter[1] << "," << parameter[0] << ")";
				break;
			case U2: 
                name << "u2(" << parameter[1] << "," << parameter[0] << ")";
				break;
			case U1: 
                name << "u1(" << parameter[0] << ")";
				break;
			case RX: 
                name << "rx(" << parameter[0] << ")";
				break;
			case RY: 
                name << "ry(" << parameter[0] << ")";
				break;
			case RZ: 
                name << "rz(" << parameter[0] << ")";
				break;
			case SWAP: 
                of << "cx " << qreg[targets[0]] << ", " << qreg[targets[1]] << ";" << std::endl;
                of << "cx " << qreg[targets[1]] << ", " << qreg[targets[0]] << ";" << std::endl;
                of << "cx " << qreg[targets[0]] << ", " << qreg[targets[1]] << ";" << std::endl;
				return;
			case iSWAP: 
                of << "cx " << qreg[targets[0]] << ", " << qreg[targets[1]] << ";" << std::endl;
                of << "cx " << qreg[targets[1]] << ", " << qreg[targets[0]] << ";" << std::endl;
                of << "cx " << qreg[targets[0]] << ", " << qreg[targets[1]] << ";" << std::endl;
				of << "s "  << qreg[targets[0]] << ";"  << std::endl;
				of << "s "  << qreg[targets[1]] << ";"  << std::endl;
				of << "h "  << qreg[targets[1]] << ";"  << std::endl;
                of << "cx " << qreg[targets[0]] << ", " << qreg[targets[1]] << ";" << std::endl;
				of << "h "  << qreg[targets[1]] << ";"  << std::endl;
				return;
			case P: 
                of << "ccx " << qreg[controls[0].qubit] << ", " << qreg[targets[1]] << ";" << std::endl;
                of << "cx " << qreg[targets[0]] << ", " << qreg[targets[1]] << ";" << std::endl;
				return;
			case Pdag: 
                of << "cx " << qreg[targets[0]] << ", " << qreg[targets[1]] << ";" << std::endl;
                of << "ccx " << qreg[controls[0].qubit] << ", " << qreg[targets[0]] << ", " << qreg[targets[1]] << ";" << std::endl;
				return;
			default: 
                std::cerr << "gate type (index) " << (int) gate << " could no be converted to qasm" << std::endl;
		}
        for(auto target: targets) 
			of << name.str() << " " << qreg[target] << ";" << std::endl;
	}

	void StandardOperation::dumpReal(std::ofstream& of) const {

	}

	void StandardOperation::dumpGRCS(std::ofstream& of) const {

	}
}