/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */
#include "QuantumComputation.hpp"

void qc::QuantumComputation::importReal(std::istream& is) {
	auto line = readRealHeader(is);
	readRealGateDescriptions(is, line);
}

int qc::QuantumComputation::readRealHeader(std::istream& is) {
	std::string cmd;
	std::string variable;
	int line = 0;

	while (true) {
		if(!static_cast<bool>(is >> cmd)) {
			throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Invalid file header");
		}
		std::transform(cmd.begin(), cmd.end(), cmd.begin(),
		        [] (unsigned char ch) { return toupper(ch); }
		        );
		++line;

		// skip comments
		if (cmd.front() == '#') {
			is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			continue;
		}

		// valid header commands start with '.'
		if (cmd.front() != '.') {
			throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Invalid file header");
		}

		if (cmd == ".BEGIN") return line; // header read complete
		else if (cmd == ".NUMVARS") {
			if(!static_cast<bool>(is >> nqubits)) {
				nqubits = 0;
			}
			nclassics = nqubits;
		} else if (cmd == ".VARIABLES") {
			for (unsigned short i = 0; i < nqubits; ++i) {
				if(!static_cast<bool>(is >> variable) || variable.at(0) == '.') {
					throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Invalid or insufficient variables declared");
				}

				qregs.insert({ variable, { i, 1 }});
				cregs.insert({ "c_" + variable, { i, 1 }});
				initialLayout.insert({ i, i });
				outputPermutation.insert({ i, i });
			}
		} else if (cmd == ".CONSTANTS") {
			is >> std::ws;
			for (unsigned short i = 0; i < nqubits; ++i) {
				const auto value = is.get();
				if (!is.good()) {
					throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Failed read in '.constants' line");
				}
				if (value == '1') {
					emplace_back<StandardOperation>(nqubits, i, X);
				} else if (value != '-' && value != '0') {
					throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Invalid value in '.constants' header: '" + std::to_string(value) + "'");
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
			std::cerr << "[WARN] File contains 'define' statement, which is currently not supported and thus simply skipped." << std::endl;
			while (cmd != ".ENDDEFINE") {
				is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				is >> cmd;
				std::transform(cmd.begin(), cmd.end(), cmd.begin(), [](const unsigned char c) { return toupper(c);});
			}
		} else {
			throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Unknown command: " + cmd);
		}

	}
}

void qc::QuantumComputation::readRealGateDescriptions(std::istream& is, int line) {
	std::regex gateRegex = std::regex("(r[xyz]|q|[0a-z](?:[+i])?)(\\d+)?(?::([-+]?[0-9]+[.]?[0-9]*(?:[eE][-+]?[0-9]+)?))?");
	std::smatch m;
	std::string cmd;

	while (!is.eof()) {
		if(!static_cast<bool>(is >> cmd)) {
			throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Failed to read command");
		}
		std::transform(cmd.begin(), cmd.end(), cmd.begin(), [](const unsigned char c) { return tolower(c);});
		++line;

		if (cmd.front() == '#') {
			is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			continue;
		}

		if (cmd == ".end") break;
		else {
			// match gate declaration
			if (!std::regex_match(cmd, m, gateRegex)) {
				throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Unsupported gate detected: " + cmd);
			}

			// extract gate information (identifier, #controls, divisor)
			OpType gate;
			if (m.str(1) == "t") { // special treatment of t(offoli) for real format
				gate = X;
			} else {
				auto it = identifierMap.find(m.str(1));
				if (it == identifierMap.end()) {
					throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Unknown gate identifier: " + m.str(1));
				}
				gate = (*it).second;
			}
			unsigned short ncontrols = m.str(2).empty() ? 0 : static_cast<unsigned short>(std::stoul(m.str(2), nullptr, 0)) - 1;
			fp lambda = m.str(3).empty() ? static_cast<fp>(0L) : static_cast<fp>(std::stold(m.str(3)));

			if (gate == V || gate == Vdag || m.str(1) == "c") ncontrols = 1;
			else if (gate == Peres || gate == Peresdag) ncontrols = 2;

			if (ncontrols >= nqubits) {
				throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Gate acts on " + std::to_string(ncontrols + 1) + " qubits, but only " + std::to_string(nqubits) + " qubits are available.");
			}

			std::string qubits, label;
			getline(is, qubits);

			std::vector<Control> controls{ };
			std::istringstream iss(qubits);

			// get controls and target
			for (int i = 0; i < ncontrols; ++i) {
				if (!(iss >> label)) {
					throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Too few variables for gate " + m.str(1));
				}

				bool negativeControl = (label.at(0) == '-');
				if (negativeControl)
					label.erase(label.begin());

				auto iter = qregs.find(label);
				if (iter == qregs.end()) {
					throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Label " + label + " not found!");
				}
				controls.emplace_back(iter->second.first, negativeControl? Control::neg: Control::pos);
			}

			if (!(iss >> label)) {
				throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Too few variables (no target) for gate " + m.str(1));
			}
			auto iter = qregs.find(label);
			if (iter == qregs.end()) {
				throw QFRException("[real parser] l:" + std::to_string(line) + " msg: Label " + label + " not found!");
			}

			updateMaxControls(ncontrols);
			unsigned short target = iter->second.first;
			unsigned short target1 = 0;
			auto x = nearbyint(lambda);
			switch (gate) {
				case None:
					throw QFRException("[real parser] l:" + std::to_string(line) + " msg: 'None' operation detected.");
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
				case RY: emplace_back<StandardOperation>(nqubits, controls, target, gate, PI / (lambda));
					break;

				case RZ:
				case Phase:
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
							emplace_back<StandardOperation>(nqubits, controls, target, gate, PI / (x));
						}
					} else {
						emplace_back<StandardOperation>(nqubits, controls, target, gate, PI / (lambda));
					}
					break;
				case SWAP:
				case Peres:
				case Peresdag:
				case iSWAP:
					target1 = controls.back().qubit;
					controls.pop_back();
					emplace_back<StandardOperation>(nqubits, controls, target, target1, gate);
					break;
				case Compound:
				case Measure:
				case Reset:
				case Snapshot:
				case ShowProbabilities:
				case Barrier:
				case ClassicControlled:
				case SX:
				case SXdag:
					std::cerr << "Operation with invalid type " << gate << " read from real file. Proceed with caution!" << std::endl;
					break;
			}
		}
	}
}
