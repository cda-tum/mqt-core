/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"

namespace qc {
	/***
     * Protected Methods
     ***/
	void QuantumComputation::importReal(std::istream& is) {
		readRealHeader(is);
		readRealGateDescriptions(is);
	}

	void QuantumComputation::readRealHeader(std::istream& is) {
		std::string cmd;
		std::string variable;

		while (true) {
			is >> cmd;
			std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::toupper);

			// skip comments
			if (cmd.front() == '#') {
				is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				continue;
			}

			// valid header commands start with '.'
			if (cmd.front() != '.') {
				std::cerr << "Invalid file header!" << std::endl;
				exit(1);
			}

			if (cmd == ".BEGIN") return; // header read complete
			else if (cmd == ".NUMVARS") {
				is >> nqubits;
				nclassics = nqubits;
			} else if (cmd == ".VARIABLES") {
				for (int i = 0; i < nqubits; ++i) {
					is >> variable;
					qregs.insert({ variable, { i, 1 }});
					cregs.insert({ "c_" + variable, { i, 1 }});
					inputPermutation.insert({ i, i });
					outputPermutation.insert({ i, i });
				}
			} else if (cmd == ".INPUTS" || cmd == ".OUTPUTS" || cmd == ".CONSTANTS" || cmd == ".GARBAGE" || cmd == ".VERSION" || cmd == ".INPUTBUS" || cmd == ".OUTPUTBUS") {
				is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				continue; // TODO: Decide if any action is really necessary here
			} else if (cmd == ".DEFINE") {
				// TODO: Defines currently not supported
				std::cerr << "Warning: File contains 'define' statement, which is currently not supported and thus simply skipped." << std::endl;
				while (cmd != ".ENDDEFINE") {
					is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
					is >> cmd;
					std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::toupper);
				}
			} else {
				std::cerr << "Unknown command: " << cmd << std::endl;
				exit(1);
			}

		}

		for (unsigned short i = 0; i < nqubits; ++i) {
			inputPermutation.insert({i, i});
			outputPermutation.insert({i,i});
		}
	}

	void QuantumComputation::readRealGateDescriptions(std::istream& is) {
		std::regex gateRegex = std::regex("(r[xyz]|q|[0a-z](?:[+i])?)(\\d+)?(?::([-+]?[0-9]+[.]?[0-9]*(?:[eE][-+]?[0-9]+)?))?");
		std::smatch m;
		std::string cmd;

		while (!is.eof()) {
			is >> cmd;
			std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);

			if (cmd.front() == '#') {
				is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				continue;
			}

			if (cmd == ".end") break;
			else {
				// match gate declaration
				if (!std::regex_match(cmd, m, gateRegex)) {
					std::cerr << "Unsupported gate detected: " << cmd << std::endl;
					exit(1);
				}

				// extract gate information (identifier, #controls, divisor)
				Gate gate;
				if (m.str(1) == "t") { // special treatment of t(offoli) for real format
					gate = X;
				} else {
					auto it = identifierMap.find(m.str(1));
					if (it == identifierMap.end()) {
						std::cerr << "Unknown gate identifier: " << m.str(1) << std::endl;
						exit(1);
					}
					gate = (*it).second;
				}
				int ncontrols = m.str(2).empty() ? 0 : std::stoi(m.str(2)) - 1;
				fp lambda = m.str(3).empty() ? 0L : std::stold(m.str(3));

				if (gate == V || gate == Vdag || m.str(1) == "c") ncontrols = 1;
				else if (gate == P || gate == Pdag) ncontrols = 2;

				if (ncontrols >= nqubits) {
					std::cerr << "Gate acts on " << ncontrols + 1 << " qubits, but only " << nqubits << " qubits are available." << std::endl;
					exit(1);
				}

				std::string qubits, label;
				getline(is, qubits);

				std::vector<Control> controls{ };
				std::istringstream iss(qubits);

				// get controls and target
				for (int i = 0; i < ncontrols; ++i) {
					if (!(iss >> label)) {
						std::cerr << "Too few variables for gate " << m.str(1) << std::endl;
						exit(1);
					}

					bool negativeControl = (label.at(0) == '-');
					if (negativeControl)
						label.erase(label.begin());

					auto iter = qregs.find(label);
					if (iter == qregs.end()) {
						std::cerr << "Label " << label << " not found!" << std::endl;
						exit(1);
					}
					controls.emplace_back(iter->second.first, negativeControl? qc::Control::neg: qc::Control::pos);
				}

				if (!(iss >> label)) {
					std::cerr << "Too few variables (no target) for gate " << m.str(1) << std::endl;
					exit(1);
				}
				auto iter = qregs.find(label);
				if (iter == qregs.end()) {
					std::cerr << "Label " << label << " not found!" << std::endl;
					exit(1);
				}

				unsigned short target = iter->second.first;
				auto x = nearbyint(lambda);
				switch (gate) {
					case None: std::cerr << "'None' operation detected." << std::endl;
						exit(1);
					case I:
					case H:
					case Y:
					case Z:
					case S:
					case Sdag:
					case T:
					case Tdag:
					case V:
					case Vdag:
					case U3:
					case U2: emplace_back<StandardOperation>(nqubits, controls, target, gate, lambda);
						break;

					case X: emplace_back<StandardOperation>(nqubits, controls, target);
						break;

					case RX:
					case RY: emplace_back<StandardOperation>(nqubits, controls, target, gate, qc::PI / (lambda));
						break;

					case RZ:
					case U1:
						if (std::abs(lambda - x) < dd::ComplexNumbers::TOLERANCE) {
							if (x == 1.0 || x == -1.0) {
								emplace_back<StandardOperation>(nqubits, controls, target, Z);
							} else if (x == 2.0) {
								emplace_back<StandardOperation>(nqubits, controls, target, S);
							} else if (x == -2.0) {
								emplace_back<StandardOperation>(nqubits, controls, target, Sdag);
							} else if (x == 4.0) {
								emplace_back<StandardOperation>(nqubits, controls, target, T);
							} else if (x == -4.0) {
								emplace_back<StandardOperation>(nqubits, controls, target, Tdag);
							} else {
								emplace_back<StandardOperation>(nqubits, controls, target, gate, qc::PI / (x));
							}
						} else {
							emplace_back<StandardOperation>(nqubits, controls, target, gate, qc::PI / (lambda));
						}
						break;
					case SWAP:
					case P:
					case Pdag: unsigned short target1 = controls.back().qubit;
						controls.pop_back();
						emplace_back<StandardOperation>(nqubits, controls, target, target1, gate);
						break;

				}
			}
		}
	}

	void QuantumComputation::importOpenQASM(std::istream& is) {
		using namespace qasm;
		// initialize parser
		Parser p(is, qregs, cregs);

		p.scan();
		p.check(Token::Kind::openqasm);
		p.check(Token::Kind::real);
		p.check(Token::Kind::semicolon);

		do {
			if (p.sym == Token::Kind::qreg) {
				p.scan();
				p.check(Token::Kind::identifier);
				std::string s = p.t.str;
				p.check(Token::Kind::lbrack);
				p.check(Token::Kind::nninteger);
				int n = p.t.val;
				p.check(Token::Kind::rbrack);
				p.check(Token::Kind::semicolon);

				p.qregs[s] = std::make_pair(nqubits, n);
				nqubits += n;
				p.nqubits = nqubits;

				// update operation descriptions
				for (auto& op: ops)
					op->setNqubits(nqubits);

			} else if (p.sym == Token::Kind::creg) {
				p.scan();
				p.check(Token::Kind::identifier);
				std::string s = p.t.str;
				p.check(Token::Kind::lbrack);
				p.check(Token::Kind::nninteger);
				int n = p.t.val;
				p.check(Token::Kind::rbrack);
				p.check(Token::Kind::semicolon);
				p.cregs[s] = std::make_pair(nclassics, n);
				nclassics += n;
			} else if (p.sym == Token::Kind::ugate || p.sym == Token::Kind::cxgate || p.sym == Token::Kind::identifier || p.sym == Token::Kind::measure || p.sym == Token::Kind::reset) {
				ops.emplace_back(p.Qop());
			} else if (p.sym == Token::Kind::gate) {
				p.GateDecl();
			} else if (p.sym == Token::Kind::include) {
				p.scan();
				p.check(Token::Kind::string);
				p.scanner->addFileInput(p.t.str);
				p.check(Token::Kind::semicolon);
			} else if (p.sym == Token::Kind::barrier) {
				p.scan();
				std::vector<std::pair<unsigned short, unsigned short>> args;
				p.ArgList(args);
				p.check(Token::Kind::semicolon);
			} else if (p.sym == Token::Kind::opaque) {
				p.OpaqueGateDecl();
			} else if (p.sym == Token::Kind::_if) {
				p.scan();
				p.check(Token::Kind::lpar);
				p.check(Token::Kind::nninteger);
				std::string creg = p.t.str;
				p.check(Token::Kind::eq);
				p.check(Token::Kind::nninteger);
				int n = p.t.val;
				p.check(Token::Kind::rpar);

				auto it = p.cregs.find(creg);
				if (it == p.cregs.end()) {
					std::cerr << "Error in if statement: " << creg << " is not a creg!" << std::endl;
				} else {
					emplace_back<ClassicControlledOperation>(p.Qop(), it->second.first + n);
				}
			} else if (p.sym == Token::Kind::snapshot) {
				p.scan();
				p.check(Token::Kind::lpar);
				p.check(Token::Kind::nninteger);
				int n = p.t.val;
				p.check(Token::Kind::rpar);

				std::vector<std::pair<unsigned short, unsigned short>> arguments;
				p.ArgList(arguments);

				p.check(Token::Kind::semicolon);

				for (auto& arg: arguments) {
					if (arg.second != 1)
						std::cerr << "ERROR in snapshot: arguments must be qubits" << std::endl;
				}

				std::vector<unsigned short> qubits;
				for (auto& arg: arguments) {
					qubits.emplace_back(arg.first);
				}

				emplace_back<NonUnitaryOperation>(nqubits, qubits, n);
			} else if (p.sym == Token::Kind::probabilities) {
				emplace_back<NonUnitaryOperation>(nqubits);
				p.scan();
				p.check(Token::Kind::semicolon);
			} else {
				std::cerr << "ERROR: unexpected statement: started with " << qasm::KindNames[p.sym] << "!" << std::endl;
				exit(1);
			}
		} while (p.sym != Token::Kind::eof);

		for (unsigned short i = 0; i < nqubits; ++i) {
			inputPermutation.insert({i, i});
			outputPermutation.insert({i,i});
		}
	}

	void QuantumComputation::importGRCS(std::istream& is, const std::string& filename) {
		size_t slash = filename.find_last_of('/');
		size_t dot = filename.find_last_of('.');
		std::string benchmark = filename.substr(slash+1, dot-slash-1);
		is >> nqubits;
		std::string line;
		std::string identifier;
		unsigned short control = 0;
		unsigned short target = 0;
		unsigned int cycle = 0;
		while (std::getline(is, line)) {
			if (line.empty()) continue;
			std::stringstream ss(line);
			ss >> cycle;
			ss >> identifier;
			if (identifier == "cz") {
				ss >> control;
				ss >> target;
				emplace_back<StandardOperation>(nqubits, Control(control), target, Z);
			} else {
				ss >> target;
				if (identifier == "h")
					emplace_back<StandardOperation>(nqubits, target, H);
				else if (identifier == "t")
					emplace_back<StandardOperation>(nqubits, target, T);
				else if (identifier == "x_1_2")
					emplace_back<StandardOperation>(nqubits, target, RX, PI_2);
				else if (identifier == "y_1_2")
					emplace_back<StandardOperation>(nqubits, target, RY, PI_2);
				else {
					std::cerr << "Unknown gate '" << identifier << "'\n";
					exit(1);
				}
			}
		}

		for (unsigned short i = 0; i < nqubits; ++i) {
			inputPermutation.insert({i, i});
			outputPermutation.insert({i,i});
		}
	}

	/***
     * Public Methods
     ***/
	unsigned long long QuantumComputation::getNindividualOps() const {
		unsigned long long nops = 0;
		for (const auto& op: ops) {
			/*
			for (int i = 0; i < op->getNqubits(); ++i) {
				if (op->getLine()[i] == 2)
					nops++;
			}
			*/
			nops += op->getTargets().size();
		}

		return nops;
	}

	void QuantumComputation::import(const std::string& filename, Format format) {
		size_t slash = filename.find_last_of('/');
		size_t dot = filename.find_last_of('.');
		name = filename.substr(slash+1, dot-slash-1);

		auto ifs = std::ifstream(filename);
		if (!ifs.good()) {
			std::cerr << "Error opening/reading from file: " << filename << std::endl;
			exit(3);
		}

		switch (format) {
			case Real: 
				importReal(ifs);
				break;
			case OpenQASM: 
				importOpenQASM(ifs);
				break;
			case GRCS: 
				importGRCS(ifs, filename);
				break;
			default: 
				ifs.close();
				std::cerr << "Format " << format << " not yet supported." << std::endl;
				exit(1);
		}
		ifs.close();
	}

	dd::Edge QuantumComputation::buildFunctionality(std::unique_ptr<dd::Package>& dd) {
		if (nqubits == 0)
			return dd->DDone;
		
		std::array<short, MAX_QUBITS> line{};
		line.fill(LINE_DEFAULT);

		dd->useMatrixNormalization(true);
		dd::Edge e = dd->makeIdent(0, nqubits-1);
		
		dd->incRef(e);

		for (auto & op : ops) {
			if (!op->isUnitary()) {
				std::cerr << "Functionality not unitary." << std::endl;
				exit(1);
			}

			auto tmp = dd->multiply(op->getDD(dd, line), e);
			dd->incRef(tmp);
			dd->decRef(e);
			e = tmp;

			dd->garbageCollect();
		}
		dd->useMatrixNormalization(false);
		return e;
	}

	dd::Edge QuantumComputation::simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) {
		// TODO: this should be part of the simulator and not of the intermediate representation
		// measurements are currently not supported here
		std::array<short, MAX_QUBITS> line{};
		line.fill(LINE_DEFAULT);

		dd::Edge e = in;
		dd->incRef(e);

		for (auto& op : ops) {
			if (!op->isUnitary()) {
				std::cerr << "Functionality not unitary." << std::endl;
				exit(1);
			}

			auto tmp = dd->multiply(op->getDD(dd, line), e);
			dd->incRef(tmp);
			dd->decRef(e);
			e = tmp;

			dd->garbageCollect();
		}

		return e;
	}

	void QuantumComputation::create_reg_array(const registerMap& regs, std::vector<std::string>& regnames, unsigned short defaultnumber, char defaultname) {
		regnames.clear();

		std::stringstream ss;
		if(regs.size() > 0) {
			for(const auto& reg: regs) {
				for(unsigned short i = 0; i < reg.second.second; i++) {
					ss << reg.first << "[" << i << "]";
					regnames.push_back(ss.str());
					ss.str(std::string());
				}
			}
		} else {
			for(unsigned short i = 0; i < defaultnumber; i++) {
				ss << defaultname << "[" << i << "]";
				regnames.push_back(ss.str());
				ss.str(std::string());
			}
		}
	}
	/*
	void QuantumComputation::compareAndEmplace(std::vector<short>& controls, unsigned short target, Gate gate, fp lambda, fp phi, fp theta) {
		if (!ops.empty()) {
			if (auto op = dynamic_cast<StandardOperation*>(ops.back().get())) {
				// TODO: only single qubit operations currently supported
				if (!op->isControlled() && controls.empty() && op->getGate() == gate &&  //TODO implement op equals gate
				     fp_equals(lambda, op->getParameter()[0]) && fp_equals(phi, op->getParameter()[1]) && fp_equals(theta, op->getParameter()[2])) {
					auto& line = op->getLine();
					if (line[target] != -1) {
						// TODO: Gate simplifications could happen here, e.g. -X-X- = -I-
					} else {
						// TODO: Explore the possibilities of combining operations here. Potentially this has a negative effect on simulation, but a positive effect on building the whoile functionality
						//line[target] = 2;
						//op->setMultiTarget(true);
						//return;
						//std::cerr << "Untouched qubit" << std::endl;
					}
				}
			}
		}

		if (gate == X)
			emplace_back<StandardOperation>(nqubits, controls, target);
		else
			emplace_back<StandardOperation>(nqubits, controls, target, gate, lambda, phi, theta);
	}
	 */

	std::ostream& QuantumComputation::print(std::ostream& os) const {
		os << std::setw(std::log10(ops.size())+5) << "i: \t\t";
		for (unsigned short i = 0; i < nqubits; ++i) {
			os << inputPermutation.at(i) << "\t";
		}
		os << std::endl;
		size_t i = 0;
		for (const auto& op:ops) {
			os << std::setw(std::log10(ops.size())+1) << ++i << ": " << *op << "\n";
		}
		os << std::setw(std::log10(ops.size())+5) << "o: \t\t";
		for (unsigned short i = 0; i < nqubits; ++i) {
			os << outputPermutation.at(i) << "\t";
		}
		os << std::endl;
		return os;
	}

	dd::Complex QuantumComputation::getEntry(std::unique_ptr<dd::Package>& dd, dd::Edge e, unsigned long long i, unsigned long long j) {
		if (dd->isTerminal(e))
			return e.w;

		dd::Complex c = dd->cn.getTempCachedComplex(1,0);
		do {
			unsigned short row = (i >> outputPermutation.at(e.p->v)) & 1;
			unsigned short col = (j >> inputPermutation.at(e.p->v)) & 1;
			e = e.p->e[dd::RADIX * row + col];
			CN::mul(c, c, e.w);
		} while (!dd->isTerminal(e));
		return c;
	}

	std::ostream& QuantumComputation::printMatrix(std::unique_ptr<dd::Package>& dd, dd::Edge e, std::ostream& os) {
		os << "Common Factor: " << e.w << "\n";
		for (unsigned long long i = 0; i < std::pow(2, nqubits); ++i) {
			for (unsigned long long j = 0; j < std::pow(2, nqubits); ++j) {
				os << std::right << std::setw(7) << std::setfill(' ') << getEntry(dd, e, i, j) << "\t";
			}
			os << std::endl;
		}
		return os;
	}

	void QuantumComputation::printBin(unsigned long long n, std::stringstream& ss) {
		if (n > 1)
			printBin(n/2, ss);
		ss << n%2;
	}

	std::ostream& QuantumComputation::printCol(std::unique_ptr<dd::Package>& dd, dd::Edge e, unsigned long long j, std::ostream& os) {
		os << "Common Factor: " << e.w << "\n";
		for (unsigned long long i = 0; i < std::pow(2, nqubits); ++i) {
			std::stringstream ss{};
			printBin(i, ss);
			os << std::setw(std::log2(std::pow(2,nqubits))) << ss.str() << ": " << getEntry(dd, e, i, j) << "\n";
		}
		return os;
	}

	std::ostream& QuantumComputation::printVector(std::unique_ptr<dd::Package>& dd, dd::Edge e, std::ostream& os) {
		return printCol(dd, e, 0, os);
	}

	std::ostream& QuantumComputation::printStatistics(std::ostream& os) {
		os << "QC Statistics:\n";
		os << "\tn: " << nqubits << std::endl;
		os << "\tm: " << ops.size() << std::endl;
		os << "--------------" << std::endl;
		return os;
	}

	void QuantumComputation::dump(const std::string& filename, Format format) {
		auto of = std::ofstream(filename);
		if (!of.good()) {
			std::cerr << "Error opening file: " << filename << std::endl;
			exit(3);
		}

		switch(format) {
			// TODO: das mit den qubit registern darf man nicht so machen. dafür am besten einmal ansehen, wie das eingelesen wird
			// Da gibt es explizit die member 'qregs' und 'cregs' in denen die richtige Information gespeichert wird
			case  OpenQASM: {
					of << "OPENQASM 2.0;"                << std::endl;
					of << "include \"qelib1.inc\";"      << std::endl;
					if(qregs.size() > 0) {
						for (auto const& qreg : qregs) {
							of << "qreg " << qreg.first << "[" << qreg.second.second << "];" << std::endl;
						}
					} else {
						of << "qreg " << DEFAULT_QREG << "[" << nqubits   << "];" << std::endl;
					}
					if(cregs.size() > 0) {
						for (auto const& creg : cregs) {
							of << "creg " << creg.first << "[" << creg.second.second << "];" << std::endl;
						}
					} else {
						of << "creg " << DEFAULT_CREG << "[" << nclassics << "];" << std::endl;
					}

					std::vector<std::string> qregnames{};
					std::vector<std::string> cregnames{};
					create_reg_array(qregs, qregnames, nqubits,   DEFAULT_QREG);
					create_reg_array(cregs, cregnames, nclassics, DEFAULT_CREG);

					for (const auto& op: ops) {
						op->dumpOpenQASM(of, qregnames, cregnames);
					}
				}
				break;
			case Real:
				std::cerr << "Dumping in real format currently not supported\n";
				break;
			case GRCS:
				std::cerr << "Dumping in GRCS format currently not supported\n";
				break;
		}
		of.close();
	}
}
