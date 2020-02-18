/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"

#include <locale>

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
			std::transform(cmd.begin(), cmd.end(), cmd.begin(),
			        [] (unsigned char ch) { return ::toupper(ch); }
			        );

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
					initialLayout.insert({ i, i });
					outputPermutation.insert({ i, i });
				}
			} else if (cmd == ".CONSTANTS") {
                is >> std::ws;
                for (int i = 0; i < nqubits; ++i) {
                    const auto value = is.get();
                    if (!is.good()) {
                        std::cerr << "[ERROR] Failed read in '.constants' line.\n";
                        std::exit(1);
                    }
                    if (value == '1') {
                        emplace_back<StandardOperation>(nqubits, i, X);
                    } else if (value != '-' && value != '0') {
                        std::cerr << "[ERROR] Invalid value in '.constants' header: '" << value << "'\n";
                        std::exit(1);
                    }
                }
                is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			} else if (cmd == ".INPUTS" || cmd == ".OUTPUTS" || cmd == ".GARBAGE" || cmd == ".VERSION" || cmd == ".INPUTBUS" || cmd == ".OUTPUTBUS") {
				// TODO .inputs: specifies initial layout (and ancillaries)
				// TODO .outputs: specifies output permutation
				// TODO .garbage: specifies garbage outputs
				is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				continue;
			} else if (cmd == ".DEFINE") {
				// TODO: Defines currently not supported
				std::cerr << "Warning: File contains 'define' statement, which is currently not supported and thus simply skipped." << std::endl;
				while (cmd != ".ENDDEFINE") {
					is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
					is >> cmd;
                    std::transform(cmd.begin(), cmd.end(), cmd.begin(), [](const unsigned char c) { return ::toupper(c);});
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
			std::transform(cmd.begin(), cmd.end(), cmd.begin(), [](const unsigned char c) { return ::tolower(c);});

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

				updateMaxControls(ncontrols);
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
					case Pdag:
					case iSWAP:
						unsigned short target1 = controls.back().qubit;
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
			} else if (p.sym == Token::Kind::ugate || p.sym == Token::Kind::cxgate || p.sym == Token::Kind::swap || p.sym == Token::Kind::identifier || p.sym == Token::Kind::measure || p.sym == Token::Kind::reset) {
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

				std::vector<unsigned short> qubits{};
				for (auto& arg: args) {
					for (unsigned short q=0; q < arg.second; ++q) {
						qubits.emplace_back(arg.first+q);
					}
				}

				emplace_back<NonUnitaryOperation>(nqubits, qubits, Barrier);
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

				std::vector<unsigned short> qubits{ };
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

	void QuantumComputation::importGRCS(std::istream& is) {
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
			initialLayout.insert({ i, i});
			outputPermutation.insert({ i, i});
		}
	}

	/***
     * Public Methods
     ***/
	unsigned long long QuantumComputation::getNindividualOps() const {
		unsigned long long nops = 0;
		for (const auto& op: ops) {
			nops += op->getTargets().size(); // TODO: this needs fixing
		}

		return nops;
	}

	void QuantumComputation::import(const std::string& filename) {
		size_t dot = filename.find_last_of('.');
		std::string extension = filename.substr(dot + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), [] (unsigned char ch) { return ::tolower(ch); });
		if (extension == "real") {
			import(filename, Real);
		} else if (extension == "qasm") {
			import(filename, OpenQASM);
		} else if(extension == "txt") {
			import(filename, GRCS);
		} else {
			std::cerr << "Extension " << extension << " not recognized." << std::endl;
			std::exit(1);
		}
	}

	void QuantumComputation::import(const std::string& filename, Format format) {
		size_t slash = filename.find_last_of('/');
		size_t dot = filename.find_last_of('.');
		name = filename.substr(slash+1, dot-slash-1);

		auto ifs = std::ifstream(filename);
		import(ifs, format);
		ifs.close();
	}

	void QuantumComputation::import(std::istream& is, Format format) {
		if (!is.good()) {
			std::cerr << "Error processing input stream: " << name << std::endl;
			exit(3);
		}

		// reset circuit before importing
		reset();

		switch (format) {
			case Real:
				importReal(is);
				break;
			case OpenQASM:
				updateMaxControls(2);
				importOpenQASM(is);
				// try to parse initial layout from qasm file
				is.clear();
				is.seekg(0, std::ios::beg);
				if (!lookForOpenQASM_IO_Layout(is)) {
					for (unsigned short i = 0; i < nqubits; ++i) {
						initialLayout.insert({ i, i});
						outputPermutation.insert({ i, i});
					}
				}
				break;
			case GRCS: importGRCS(is);
				break;
			default:
				std::cerr << "Format " << format << " not yet supported." << std::endl;
				exit(1);
		}
	}

	void QuantumComputation::addQubitRegister(unsigned short nq, const char* reg_name) {
		if (nqubits + nancillae + nq > dd::MAXN) {
			std::cerr << "Adding additional qubits results in too many qubits. " << nqubits + nancillae + nq << " vs. " << dd::MAXN << std::endl;
			exit(1);
		}

		if (qregs.count(reg_name)) {
			auto& reg = qregs.at(reg_name);
			if(reg.first+reg.second == nqubits+nancillae) {
				reg.second+=nq;
			} else {
				std::cerr << "Augmenting existing qubit registers is only supported for the last register in a circuit" << std::endl;
				exit(1);
			}
		} else {
			qregs.insert({reg_name, {nqubits, nq}});
		}
		assert(nancillae == 0); // should only reach this point if no ancillae are present

		for (unsigned short i = 0; i < nq; ++i) {
			unsigned short j = nqubits + i;
			initialLayout.insert({ j, j});
			outputPermutation.insert({ j, j});
		}
		nqubits += nq;

		for (auto& op:ops) {
			op->setNqubits(nqubits + nancillae);
		}
	}

	void QuantumComputation::addClassicalRegister(unsigned short nc, const char* reg_name) {

		if (cregs.count(reg_name)) {
			std::cerr << "Augmenting existing classical registers is currently not supported" << std::endl;
			exit(1);
		}

		cregs.insert({reg_name, {nclassics, nc}});
		nclassics += nc;
	}

	void QuantumComputation::addAncillaryRegister(unsigned short nq, const char* reg_name) {
		if (nqubits + nancillae + nq > dd::MAXN) {
			std::cerr << "Adding additional ancillaries results in too many qubits. " << nqubits + nancillae + nq << " vs. " << dd::MAXN << std::endl;
			exit(1);
		}

		unsigned short totalqubits = nqubits + nancillae;

		if (ancregs.count(reg_name)) {
			auto& reg = ancregs.at(reg_name);
			if(reg.first+reg.second == totalqubits) {
				reg.second+=nq;
			} else {
				std::cerr << "Augmenting existing ancillary registers is only supported for the last register in a circuit" << std::endl;
				exit(1);
			}
		} else {
			ancregs.insert({reg_name, {totalqubits, nq}});
		}

		for (unsigned short i = 0; i < nq; ++i) {
			unsigned short j = totalqubits + i;
			initialLayout.insert({ j, j});
			outputPermutation.insert({ j, j});
		}
		nancillae += nq;

		for (auto& op:ops) {
			op->setNqubits(nqubits + nancillae);
		}
	}

	// removes the i-th logical qubit and returns the index j it was assigned to in the initial layout
	// i.e., initialLayout[j] = i
	std::pair<unsigned short, short> QuantumComputation::removeQubit(unsigned short logical_qubit_index) {
		#if DEBUG_MODE_QC
		std::cout << "Trying to remove logical qubit: " << logical_qubit_index << std::endl;
		#endif

		// Find index of the physical qubit i is assigned to
		auto physical_qubit_index = 0;
		for (const auto& Q:initialLayout) {
			if (Q.second == logical_qubit_index)
				physical_qubit_index = Q.first;
		}

		#if DEBUG_MODE_QC
		std::cout << "Found index " << logical_qubit_index << " is assigned to: " << physical_qubit_index << std::endl;
		printRegisters(std::cout);
		#endif

		// get register and register-index of the corresponding qubit
		auto reg = getQubitRegisterAndIndex(physical_qubit_index);

		#if DEBUG_MODE_QC
		std::cout << "Found register: " << reg.first << ", and index: " << reg.second << std::endl;
		printRegisters(std::cout);
		#endif

		if (isAncilla(physical_qubit_index)) {
			#if DEBUG_MODE_QC
			std::cout << physical_qubit_index << " is ancilla" << std::endl;
			#endif
			// first index
			if (reg.second == 0) {
				// last remaining qubit of register
				if (ancregs[reg.first].second == 1) {
					// delete register
					ancregs.erase(reg.first);
				}
				// first qubit of register
				else {
					ancregs[reg.first].first++;
					ancregs[reg.first].second--;
				}
			// last index
			} else if (reg.second == ancregs[reg.first].second-1) {
				// reduce count of register
				ancregs[reg.first].second--;
			} else {
				auto ancreg = ancregs.at(reg.first);
				auto low_part = reg.first + "_l";
				auto low_index = ancreg.first;
				auto low_count = reg.second;
				auto high_part = reg.first + "_h";
				auto high_index = ancreg.first + reg.second + 1;
				auto high_count = ancreg.second - reg.second - 1;

				#if DEBUG_MODE_QC
				std::cout << "Splitting register: " << reg.first << ", into:" << std::endl;
				std::cout << low_part << ": {" << low_index << ", " << low_count << "}" << std::endl;
				std::cout << high_part << ": {" << high_index << ", " << high_count << "}" << std::endl;
				#endif

				ancregs.erase(reg.first);
				ancregs.insert({low_part, {low_index, low_count}});
				ancregs.insert({high_part, {high_index, high_count}});
			}
			// reduce ancilla count
			nancillae--;
		} else {
			if (reg.second == 0) {
				// last remaining qubit of register
				if (qregs[reg.first].second == 1) {
					// delete register
					qregs.erase(reg.first);
				}
					// first qubit of register
				else {
					qregs[reg.first].first++;
					qregs[reg.first].second--;
				}
			// last index
			} else if (reg.second == qregs[reg.first].second-1) {
				// reduce count of register
				qregs[reg.first].second--;
			} else {
				auto qreg = qregs.at(reg.first);
				auto low_part = reg.first + "_l";
				auto low_index = qreg.first;
				auto low_count = reg.second;
				auto high_part = reg.first + "_h";
				auto high_index = qreg.first + reg.second + 1;
				auto high_count = qreg.second - reg.second - 1;

				#if DEBUG_MODE_QC
				std::cout << "Splitting register: " << reg.first << ", into:" << std::endl;
				std::cout << low_part << ": {" << low_index << ", " << low_count << "}" << std::endl;
				std::cout << high_part << ": {" << high_index << ", " << high_count << "}" << std::endl;
				#endif

				qregs.erase(reg.first);
				qregs.insert({low_part, {low_index, low_count}});
				qregs.insert({high_part, {high_index, high_count}});
			}
			// reduce qubit count
			nqubits--;
		}

		#if DEBUG_MODE_QC
		std::cout << "Updated registers: " << std::endl;
		printRegisters(std::cout);
		std::cout << "nqubits: " << nqubits << ", nancillae: " << nancillae << std::endl;
		#endif

		// adjust initial layout permutation
		initialLayout.erase(physical_qubit_index);

		#if DEBUG_MODE_QC
		std::cout << "Updated initial layout: " << std::endl;
		printPermutationMap(initialLayout, std::cout);
		#endif

		// remove potential output permutation entry
		short output_qubit_index = -1;
		auto it = outputPermutation.find(physical_qubit_index);
		if (it != outputPermutation.end()) {
			output_qubit_index = it->second;
			// erasing entry
			outputPermutation.erase(physical_qubit_index);
			#if DEBUG_MODE_QC
			std::cout << "Updated output permutation: " << std::endl;
			printPermutationMap(outputPermutation, std::cout);
			#endif
		}

		// update all operations
		for (auto& op:ops) {
			op->setNqubits(nqubits + nancillae);
		}

		return { physical_qubit_index, output_qubit_index};
	}

	// adds j-th physical qubit as ancilla to the end of reg or creates the register if necessary
	void QuantumComputation::addAncillaryQubit(unsigned short physical_qubit_index, short output_qubit_index) {
		if(initialLayout.count(physical_qubit_index) || outputPermutation.count(physical_qubit_index)) {
			std::cerr << "Attempting to insert physical qubit that is already assigned" << std::endl;
			exit(1);
		}

		#if DEBUG_MODE_QC
		std::cout << "Trying to add physical qubit " << physical_qubit_index
				  << " as ancillary with output qubit index: " << output_qubit_index << std::endl;
		#endif

		// index of logical qubit
		unsigned short logical_qubit_index = 0;
		// TODO: here it could be checked if an ancillary register already exists in the surrounding of physical_qubit_index and then it could be fused
		auto new_reg_name = std::string(DEFAULT_ANCREG) + "_" + std::to_string(physical_qubit_index);

		// check if register already exists
		if(ancregs.count(new_reg_name)) {
			std::cerr << "Register " << new_reg_name << " should not exist" << std::endl;
			exit(1);
		} else {
			// create new register
			ancregs.insert({new_reg_name, { physical_qubit_index, 1}});
			logical_qubit_index = nqubits + nancillae;

		}
		// increase ancillae count
		nancillae++;

		#if DEBUG_MODE_QC
		std::cout << "Updated registers: " << std::endl;
		printRegisters(std::cout);
		std::cout << "nqubits: " << nqubits << ", nancillae: " << nancillae << std::endl;
		#endif

		// adjust initial layout
		initialLayout.insert({ physical_qubit_index, logical_qubit_index});
		#if DEBUG_MODE_QC
		std::cout << "Updated initial layout: " << std::endl;
		printPermutationMap(initialLayout, std::cout);
		#endif

		// adjust output permutation
		if (output_qubit_index >= 0) {
			outputPermutation.insert({ physical_qubit_index, output_qubit_index});
			#if DEBUG_MODE_QC
			std::cout << "Updated output permutation: " << std::endl;
			printPermutationMap(outputPermutation, std::cout);
			#endif
		}

		// update all operations
		for (auto& op:ops) {
			op->setNqubits(nqubits + nancillae);
		}
	}

	void QuantumComputation::addQubit(unsigned short logical_qubit_index, unsigned short physical_qubit_index, short output_qubit_index) {
		if (initialLayout.count(physical_qubit_index) || outputPermutation.count(physical_qubit_index)) {
			std::cerr << "Attempting to insert physical qubit that is already assigned" << std::endl;
			exit(1);
		}

		if (logical_qubit_index > nqubits) {
			std::cerr << "There are currently only " << nqubits << " qubits in the circuit. Adding "
					  << logical_qubit_index << " is therefore not possible at the moment." << std::endl;
			exit(1);
			// TODO: this does not necessarily have to lead to an error. A new qubit register could be created and all ancillaries shifted
		}

		std::cerr << "Function 'addQubit' currently not implemented" << std::endl;
		exit(1);
		// TODO: implement this function

		// qubit either fits in the beginning/the end or in between two existing registers
		// if it fits at the very end of the last qubit register it has to be checked
		// if the indices of ancillary registers need to be shifted by one

		// increase qubit count
		nqubits++;
		// adjust initial layout
		initialLayout.insert({ physical_qubit_index, logical_qubit_index});
		if (output_qubit_index >= 0) {
			// adjust output permutation
			outputPermutation.insert({physical_qubit_index, output_qubit_index});
		}
		// update all operations
		for (auto& op:ops) {
			op->setNqubits(nqubits + nancillae);
		}
	}


	void QuantumComputation::reduceAncillae(dd::Edge& e, std::unique_ptr<dd::Package>& dd) {
		if (e.p->v < nqubits) return;
		for(auto& edge: e.p->e)
			reduceAncillae(edge, dd);

		auto saved = e;
		e = dd->makeNonterminal(e.p->v, {e.p->e[0], dd::Package::DDzero, e.p->e[2], dd::Package::DDzero });
		dd->incRef(e);
		dd->decRef(saved);
		dd->garbageCollect();
	}

	void QuantumComputation::reduceGarbage(dd::Edge& e, std::unique_ptr<dd::Package>& dd) {
		if (e.p->v < nqubits) return;
		for(auto& edge: e.p->e)
			reduceGarbage(edge, dd);

		auto saved = e;
		e = dd->makeNonterminal(e.p->v, { dd->add(e.p->e[0], e.p->e[2]), dd->add(e.p->e[1], e.p->e[3]), dd::Package::DDzero, dd::Package::DDzero });
		dd->incRef(e);
		dd->decRef(saved);
	}


	dd::Edge QuantumComputation::createInitialMatrix(std::unique_ptr<dd::Package>& dd) {
		dd::Edge e = dd->makeIdent(0, short(nqubits+nancillae-1));
		dd->incRef(e);
		reduceAncillae(e, dd);
		return e;
	}


	dd::Edge QuantumComputation::buildFunctionality(std::unique_ptr<dd::Package>& dd) {
		if (nqubits + nancillae == 0)
			return dd->DDone;
		
		std::array<short, MAX_QUBITS> line{};
		line.fill(LINE_DEFAULT);
		permutationMap map = initialLayout;

		dd->useMatrixNormalization(true);
		dd::Edge e = createInitialMatrix(dd);
		dd->incRef(e);

		for (auto & op : ops) {
			if (!op->isUnitary()) {
				std::cerr << "Functionality not unitary." << std::endl;
				exit(1);
			}

			auto tmp = dd->multiply(op->getDD(dd, line, map), e);

			dd->incRef(tmp);
			dd->decRef(e);
			e = tmp;

			dd->garbageCollect();
		}
		dd->useMatrixNormalization(false);
		return e;
	}

	dd::Edge QuantumComputation::simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) {
		// measurements are currently not supported here
		std::array<short, MAX_QUBITS> line{};
		line.fill(LINE_DEFAULT);
		permutationMap map = initialLayout;

		dd::Edge e = in;
		dd->incRef(e);

		for (auto& op : ops) {
			if (!op->isUnitary()) {
				std::cerr << "Functionality not unitary." << std::endl;
				exit(1);
			}

			auto tmp = dd->multiply(op->getDD(dd, line, map), e);

			dd->incRef(tmp);
			dd->decRef(e);
			e = tmp;

			dd->garbageCollect();
		}

		return e;
	}

	void QuantumComputation::create_reg_array(const registerMap& regs, regnames_t& regnames, unsigned short defaultnumber, const char* defaultname) {
		regnames.clear();

		std::stringstream ss;
		if(!regs.empty()) {
			for(const auto& reg: regs) {
				for(unsigned short i = 0; i < reg.second.second; i++) {
					ss << reg.first << "[" << i << "]";
					regnames.push_back(std::make_pair(reg.first, ss.str()));
					ss.str(std::string());
				}
			}
		} else {
			for(unsigned short i = 0; i < defaultnumber; i++) {
				ss << defaultname << "[" << i << "]";
				regnames.push_back(std::make_pair(defaultname, ss.str()));
				ss.str(std::string());
			}
		}
	}

	std::ostream& QuantumComputation::print(std::ostream& os) const {
		os << std::setw((int)std::log10(ops.size())+1) << "i" << ": \t\t\t";
		for (const auto& Q: initialLayout) {
			os << Q.second << "\t";
		}
		/*for (unsigned short i = 0; i < nqubits + nancillae; ++i) {
			auto it = initialLayout.find(i);
			if(it == initialLayout.end()) {
				os << "|\t";
			} else {
				os << it->second << "\t";
			}
		}*/
		os << std::endl;
		size_t i = 0;
		for (const auto& op:ops) {
			os << std::setw((int)std::log10(ops.size())+1) << ++i << ": \t";
			op->print(os, initialLayout);
			os << std::endl;
		}
		os << std::setw((int)std::log10(ops.size())+1) << "o" << ": \t\t\t";
		for(const auto& physical_qubit: initialLayout) {
			auto it = outputPermutation.find(physical_qubit.first);
			if(it == outputPermutation.end()) {
				os << "|\t";
			} else {
				os << it->second << "\t";
			}
		}
		os << std::endl;
		return os;
	}

	dd::Complex QuantumComputation::getEntry(std::unique_ptr<dd::Package>& dd, dd::Edge e, unsigned long long i, unsigned long long j) {
		if (dd->isTerminal(e))
			return e.w;

		dd::Complex c = dd->cn.getTempCachedComplex(1,0);
		do {
			unsigned short row = (i >> outputPermutation.at(e.p->v)) & 1u;
			unsigned short col = (j >> initialLayout.at(e.p->v)) & 1u;
			e = e.p->e[dd::RADIX * row + col];
			CN::mul(c, c, e.w);
		} while (!dd->isTerminal(e));
		return c;
	}

	std::ostream& QuantumComputation::printMatrix(std::unique_ptr<dd::Package>& dd, dd::Edge e, std::ostream& os) {
		os << "Common Factor: " << e.w << "\n";
		for (unsigned long long i = 0; i < (1ull << (unsigned int)(nqubits+nancillae)); ++i) {
			for (unsigned long long j = 0; j < (1ull << (unsigned int)(nqubits+nancillae)); ++j) {
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
		for (unsigned long long i = 0; i < (1ull << (unsigned int)(nqubits+nancillae)); ++i) {
			std::stringstream ss{};
			printBin(i, ss);
			os << std::setw(nqubits + nancillae) << ss.str() << ": " << getEntry(dd, e, i, j) << "\n";
		}
		return os;
	}

	std::ostream& QuantumComputation::printVector(std::unique_ptr<dd::Package>& dd, dd::Edge e, std::ostream& os) {
		return printCol(dd, e, 0, os);
	}

	std::ostream& QuantumComputation::printStatistics(std::ostream& os) {
		os << "QC Statistics:\n";
		os << "\tn: " << nqubits << std::endl;
		os << "\tanc: " << nancillae << std::endl;
		os << "\tm: " << ops.size() << std::endl;
		os << "--------------" << std::endl;
		return os;
	}

	void QuantumComputation::dump(const std::string& filename) {
		size_t dot = filename.find_last_of('.');
		std::string extension = filename.substr(dot + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return ::tolower(c); });
		if (extension == "real") {
			dump(filename, Real);
		} else if (extension == "qasm") {
			dump(filename, OpenQASM);
		} else if(extension == "py") {
			dump(filename, Qiskit);
		} else {
			std::cerr << "Extension " << extension << " not recognized/supported for dumping." << std::endl;
			std::exit(1);
		}
	}

	void QuantumComputation::dump(const std::string& filename, Format format) {
		auto of = std::ofstream(filename);
		if (!of.good()) {
			std::cerr << "Error opening file: " << filename << std::endl;
			exit(3);
		}

		switch(format) {
			case  OpenQASM: {
					of << "OPENQASM 2.0;"                << std::endl;
					of << "include \"qelib1.inc\";"      << std::endl;
					if(!qregs.empty()) {
						for (auto const& qreg : qregs) {
							of << "qreg " << qreg.first << "[" << qreg.second.second << "];" << std::endl;
						}
					} else if (nqubits > 0) {
						of << "qreg " << DEFAULT_QREG << "[" << nqubits   << "];" << std::endl;
					}
					if(!cregs.empty()) {
						for (auto const& creg : cregs) {
							of << "creg " << creg.first << "[" << creg.second.second << "];" << std::endl;
						}
					} else if (nclassics > 0) {
						of << "creg " << DEFAULT_CREG << "[" << nclassics << "];" << std::endl;
					}
					if(!ancregs.empty()) {
						for (auto const& ancreg : ancregs) {
							of << "qreg " << ancreg.first << "[" << ancreg.second.second << "];" << std::endl;
						}
					} else if (nancillae > 0) {
						of << "qreg " << DEFAULT_ANCREG << "[" << nancillae << "];" << std::endl;
					}

					regnames_t qregnames{};
					regnames_t cregnames{};
					regnames_t ancregnames{};

					create_reg_array(qregs, qregnames, nqubits, DEFAULT_QREG);
					create_reg_array(cregs, cregnames, nclassics, DEFAULT_CREG);
					create_reg_array(ancregs, ancregnames, nancillae, DEFAULT_ANCREG);
					for (auto& ancregname: ancregnames)
						qregnames.push_back(ancregname);

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
			case Qiskit:
				unsigned short totalQubits = nqubits + nancillae + (max_controls >= 2? max_controls-2: 0);
				if (totalQubits > 53) {
					std::cerr << "No more than 53 total qubits are currently supported" << std::endl;
					break;
				}

				// For the moment all registers are fused together into for simplicity
				// This may be adapted in the future
				of << "from qiskit import *" << std::endl;
				of << "from qiskit.test.mock import ";
				unsigned short narchitecture = 0;
				if (totalQubits <= 5) {
					of << "FakeBurlington";
					narchitecture = 5;
				} else if (totalQubits <= 20) {
					of << "FakeBoeblingen";
					narchitecture = 20;
				} else if (totalQubits <= 53) {
					of << "FakeRochester";
					narchitecture = 53;
				}
				of << std::endl;
				of << "from qiskit.converters import circuit_to_dag, dag_to_circuit" << std::endl;
				of << "from qiskit.transpiler.passes import *" << std::endl;
				of << "from math import pi" << std::endl << std::endl;

				of << DEFAULT_QREG << " = QuantumRegister(" << nqubits << ", '" << DEFAULT_QREG << "')" << std::endl;
				if (nclassics > 0) {
					of << DEFAULT_CREG << " = ClassicalRegister(" << nclassics << ", '" << DEFAULT_CREG << "')" << std::endl;
				}
				if (nancillae > 0) {
					of << DEFAULT_ANCREG << " = QuantumRegister(" << nancillae << ", '" << DEFAULT_ANCREG << "')" << std::endl;
				}
				if (max_controls > 2) {
					of << DEFAULT_MCTREG << " = QuantumRegister(" << max_controls - 2 << ", '"<< DEFAULT_MCTREG << "')" << std::endl;
				}
				of << "qc = QuantumCircuit(";
				of << DEFAULT_QREG;
				if (nclassics > 0) {
					of << ", " << DEFAULT_CREG;
				}
				if (nancillae > 0) {
					of << ", " << DEFAULT_ANCREG;
				}
				if(max_controls > 2) {
					of << ", " << DEFAULT_MCTREG;
				}
				of << ")" << std::endl << std::endl;

				regnames_t qregnames{};
				regnames_t cregnames{};
				regnames_t ancregnames{};
				create_reg_array({}, qregnames, nqubits, DEFAULT_QREG);
				create_reg_array({}, cregnames, nclassics, DEFAULT_CREG);
				create_reg_array({}, ancregnames, nancillae, DEFAULT_ANCREG);

				for (auto& ancregname: ancregnames)
					qregnames.push_back(ancregname);

				for (const auto& op: ops) {
					op->dumpQiskit(of, qregnames, cregnames, DEFAULT_MCTREG);
				}
				// add measurement for determining output mapping
				of << "qc.measure_all()" << std::endl;

				of << "qc_transpiled = transpile(qc, backend=";
				if (totalQubits <= 5) {
					of << "FakeBurlington";
				} else if (totalQubits <= 20) {
					of << "FakeBoeblingen";
				} else if (totalQubits <= 53) {
					of << "FakeRochester";
				}
				of << "(), optimization_level=1)" << std::endl << std::endl;
				of << "layout = qc_transpiled._layout" << std::endl;
				of << "virtual_bits = layout.get_virtual_bits()" << std::endl;

				of << "f = open(\"" << filename.substr(0, filename.size()-3) << R"(_transpiled.qasm", "w"))" << std::endl;
				of << R"(f.write("// i"))" << std::endl;
				of << "for qubit in " << DEFAULT_QREG << ":" << std::endl;
				of << '\t' << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
				if (nancillae > 0) {
					of << "for qubit in " << DEFAULT_ANCREG << ":" << std::endl;
					of << '\t' << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
				}
				if (max_controls > 2) {
					of << "for qubit in " << DEFAULT_MCTREG << ":" << std::endl;
					of << '\t' << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
				}
				if (totalQubits < narchitecture) {
					of << "for reg in layout.get_registers():" << std::endl;
					of << '\t' << "if reg.name is 'ancilla':" << std::endl;
					of << "\t\t" << "for qubit in reg:" << std::endl;
					of << "\t\t\t" << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
				}
				of << R"(f.write("\n"))" << std::endl;
				of << "dag = circuit_to_dag(qc_transpiled)" << std::endl;
				of << "out = [item for sublist in list(dag.layers())[-1]['partition'] for item in sublist]" << std::endl;
				of << R"(f.write("// o"))" << std::endl;
				of << "for qubit in out:" << std::endl;
				of << '\t' << R"(f.write(" " + str(qubit.index)))" << std::endl;
				of << R"(f.write("\n"))" << std::endl;
				// remove measurements again
				of << "qc_transpiled = dag_to_circuit(RemoveFinalMeasurements().run(dag))" << std::endl;
				of << "f.write(qc_transpiled.qasm())" << std::endl;
				of << "f.close()" << std::endl;
				break;
		}
		of.close();
	}

	bool QuantumComputation::isIdleQubit(unsigned short i) {
		for(const auto& op:ops) {
			if (op->actsOn(i))
				return false;
		}
		return true;
	}

	void QuantumComputation::stripTrailingIdleQubits() {
		for (int physical_qubit_index= nqubits + nancillae - 1; physical_qubit_index >= 0; physical_qubit_index--) {
			if(isIdleQubit(physical_qubit_index)) {
				unsigned short logical_qubit_index = initialLayout[physical_qubit_index];
				#if DEBUG_MODE_QC
				std::cout << "Trying to strip away idle qubit: " << physical_qubit_index
						  << ", which corresponds to logical qubit: " << logical_qubit_index << std::endl;
				print(std::cout);
				#endif
				auto check = removeQubit(logical_qubit_index);
				assert(check.first == physical_qubit_index);

				#if DEBUG_MODE_QC
				std::cout << "Resulting in: " << std::endl;
				print(std::cout);
				#endif
			} else {
				break;
				// TODO: in principle it should be possible to remove qubits not only from the end, but also somewhere in the middle
				// However this currently results in some errors (4gt11_84, 4gt11-v1_85, dk27_225, sqr6_259) from O0-O3
			}
		}
		for(auto& op:ops) {
			op->setNqubits(nqubits + nancillae);
		}
	}

	std::string QuantumComputation::getQubitRegister(unsigned short physical_qubit_index) {

		for (const auto& reg:qregs) {
			unsigned short start_idx = reg.second.first;
			unsigned short count = reg.second.second;
			if (physical_qubit_index < start_idx) continue;
			if (physical_qubit_index >= start_idx + count) continue;
			return reg.first;
		}
		for (const auto& reg:ancregs) {
			unsigned short start_idx = reg.second.first;
			unsigned short count = reg.second.second;
			if (physical_qubit_index < start_idx) continue;
			if (physical_qubit_index >= start_idx + count) continue;
			return reg.first;
		}

		std::cerr << "Qubit index " << physical_qubit_index << " not found in any register" << std::endl;
		exit(1);
	}

	std::pair<std::string, unsigned short> QuantumComputation::getQubitRegisterAndIndex(unsigned short physical_qubit_index) {
		std::string reg_name = getQubitRegister(physical_qubit_index);
		unsigned short index = 0;
		auto it = qregs.find(reg_name);
		if (it != qregs.end()) {
			index = physical_qubit_index - it->second.first;
		} else {
			auto it_anc = ancregs.find(reg_name);
			if (it_anc != ancregs.end()) {
				index = physical_qubit_index - it_anc->second.first;
			}
			// no else branch needed here, since error would have already shown in getQubitRegister(physical_qubit_index)
		}
		return {reg_name, index};
	}

	std::ostream& QuantumComputation::printPermutationMap(const permutationMap &map, std::ostream &os) {
		for(const auto& Q: map) {
			os <<"\t" << Q.first << ": " << Q.second << std::endl;
		}
		return os;
	}

	std::ostream& QuantumComputation::printRegisters(std::ostream& os) {
		os << "qregs:";
		for(const auto& qreg: qregs) {
			os << " {" << qreg.first << ", {" << qreg.second.first << ", " << qreg.second.second << "}}";
		}
		os << std::endl;
		if (!ancregs.empty()) {
			os << "ancregs:";
			for(const auto& ancreg: ancregs) {
				os << " {" << ancreg.first <<", {" << ancreg.second.first << ", " << ancreg.second.second << "}}";
			}
			os << std::endl;
		}
		os << "cregs:";
		for(const auto& creg: cregs) {
			os << " {" << creg.first <<", {" << creg.second.first << ", " << creg.second.second << "}}";
		}
		os << std::endl;
		return os;
	}

	bool QuantumComputation::lookForOpenQASM_IO_Layout(std::istream& ifs) {
		std::string line;
		while (std::getline(ifs,line)) {
			/*
			 * check all comment lines in header for layout information in the following form:
			        // i Q0 Q1 ... Qn
					// o q0 q1 ... qn
		        where i describes the initial layout, e.g. 'i 2 1 0' means q2 -> Q0, q1 -> Q1, q0 -> Q2
		        and j describes the output permutation, e.g. 'o 2 1 0' means q0 -> Q2, q1 -> Q1, q2 -> Q0
			 */
			if(line.rfind("//", 0) == 0) {
				if (line.find('i') != std::string::npos) {
					unsigned short i = 0;
					auto ss = std::stringstream(line.substr(4));
					for (unsigned short j=0; j < getNqubits(); ++j) {
						if (!(ss >> i)) return false;
						initialLayout.insert({i, j});
					}
				} else if (line.find('o') != std::string::npos) {
					unsigned short j = 0;
					auto ss = std::stringstream(line.substr(4));
					for (unsigned short i=0; i < getNqubits(); ++i) {
						if (!(ss >> j)) return true; // allow for incomplete output permutation
						outputPermutation.insert({j, i});
					}
					return true;
				}
			}
		}
		return false;
	}

	void QuantumComputation::fuseCXtoSwap() {
		// search for
		//      cx a b
		//      cx b a      (could also be mct ... b a in general)
		//      cx a b
		for (auto it=ops.begin(); it != ops.end(); ++it ) {
			auto op0 = dynamic_cast<StandardOperation*>((*it).get());
			if (op0) {
				// search for CX a b
				if (op0->getGate() == X &&
				    op0->getControls().size() == 1 &&
				    op0->getControls().at(0).type == Control::pos) {

					unsigned short control = op0->getControls().at(0).qubit;
					assert(op0->getTargets().size() == 1);
					unsigned short target = op0->getTargets().at(0);

					auto it1 = it;
					it1++;
					if(it1 != ops.end()) {
						auto op1 = dynamic_cast<StandardOperation*>((*it1).get());
						if (op1) {
							// search for CX b a (or mct ... b a)
							if (op1->getGate() == X &&
							    op1->getTargets().at(0) == control &&
							    std::any_of(op1->getControls().begin(), op1->getControls().end(), [target](qc::Control c)  -> bool {return c.qubit == target;})) {
								assert(op1->getTargets().size() == 1);

								auto it2 = it1;
								it2++;
								if (it2 != ops.end()) {
									auto op2 = dynamic_cast<StandardOperation*>((*it2).get());
									if (op2) {
										// search for CX a b
										if (op2->getGate() == X &&
										    op2->getTargets().at(0) == target &&
										    op2->getControls().size() == 1 &&
										    op2->getControls().at(0).qubit == control) {

											assert(op2->getTargets().size() == 1);

											op0->setGate(SWAP);
											op0->setTargets({target, control});
											op0->setControls({});
											for (const auto& c: op1->getControls()) {
												if (c.qubit != target)
													op0->getControls().push_back(c);
											}
											ops.erase(it2);
											ops.erase(it1);
										} else {
											//continue;
											// try replacing
											//  CX a b
											//  CX b a
											// with
											//  SWAP a b
											//  CX   a b
											// in order to enable more efficient swap reduction
											if(op1->getControls().size() != 1) continue;

											op0->setGate(SWAP);
											op0->setTargets({target, control});
											op0->setControls({});

											op1->setTargets({target});
											op1->setControls({Control(control)});
										}
									}
								} else {
									if(op1->getControls().size() != 1) continue;

									op0->setGate(SWAP);
									op0->setTargets({target, control});
									op0->setControls({});

									op1->setTargets({target});
									op1->setControls({Control(control)});
								}
							}
						}
					}
				}
			}
		}
	}

	unsigned short QuantumComputation::getHighestLogicalQubitIndex() {
		unsigned short max_index = 0;
		for (const auto& physical_qubit: initialLayout) {
			if (physical_qubit.second > max_index) {
				max_index = physical_qubit.second;
			}
		}
		return max_index;
	}

	bool QuantumComputation::isAncilla(unsigned short i) {
		for (const auto& ancreg: ancregs) {
			if (ancreg.second.first <= i && i < ancreg.second.first+ancreg.second.second)
				return true;
		}
		return false;
	}
}
