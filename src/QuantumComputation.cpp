//
// Created by Lukas Burgholzer on 25.09.19.
//
#include "QuantumComputation.h"

#include <algorithm>
#include <regex>
#include <limits>

namespace qc {

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
					qregs.insert({ variable, { i, i + 1 }});
					cregs.insert({ "c_" + variable, { i, i + 1 }});
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

				std::vector<short> controls{ };
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
					controls.push_back(negativeControl ? -(iter->second.first) : iter->second.first);
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
					case Pdag: unsigned short target1 = controls.back();
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
	}

	dd::Edge QuantumComputation::buildFunctionality(std::unique_ptr<dd::Package>& dd, int nops) {
		if (nops == -1)
			nops= ops.size();
		dd->useMatrixNormalization(true);
		dd::Edge e = dd->makeIdent(0, nqubits-1);
		dd->incRef(e);

		for (int i=0; i < nops; ++i) {
			if (!ops[i]->isUnitary()) {
				std::cerr << "Functionality not unitary." << std::endl;
				exit(1);
			}

			auto tmp = dd->multiply(ops[i]->getDD(dd), e);
			dd->incRef(tmp);
			dd->decRef(e);
			e = tmp;

			dd->garbageCollect();
			//std::cout << dd->size(e) << std::endl;
		}
		//dd->useMatrixNormalization(false);
		return e;
	}

	dd::Edge QuantumComputation::simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) {
		dd::Edge e = in;
		dd->incRef(e);

		for (auto& op : ops) {
			if (!op->isUnitary()) {
				std::cerr << "Functionality not unitary." << std::endl;
				exit(1);
			}

			auto tmp = dd->multiply(op->getDD(dd), e);
			dd->incRef(tmp);
			dd->decRef(e);
			e = tmp;

			dd->garbageCollect();
		}

		return e;
	}

	void QuantumComputation::compareAndEmplace(std::vector<short>& controls, unsigned short target, Gate gate, fp lambda, fp phi, fp theta) {
		if (!ops.empty()) {
			if (auto op = dynamic_cast<StandardOperation*>(ops.back().get())) {
				// TODO: only single qubit operations currently supported
				if (!op->isControlled() && controls.empty() && op->getGate() == gate && fp_equals(lambda, op->getParameter()[0]) && fp_equals(phi, op->getParameter()[1]) && fp_equals(theta, op->getParameter()[2])) {
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

		emplace_back<StandardOperation>(nqubits, controls, target, gate, lambda, phi, theta);
	}
}
