/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */
#include "QuantumComputation.hpp"

void qc::QuantumComputation::importTFC(std::istream& is) {
	std::map<std::string, unsigned short> varMap{};
	auto line = readTFCHeader(is, varMap);
	readTFCGateDescriptions(is, line, varMap);
}

int qc::QuantumComputation::readTFCHeader(std::istream& is, std::map<std::string, unsigned short>& varMap) {
	std::string cmd;
	std::string variable;
	std::string identifier;
	int line = 0;

	std::string delimiter = ",";
	size_t pos = 0;

	std::vector<std::string> variables{};
	std::vector<std::string> inputs{};
	std::vector<std::string> outputs{};
	std::vector<std::string> constants{};

	while (true) {
		if(!static_cast<bool>(is >> cmd)) {
			throw QFRException("[tfc parser] l:" + std::to_string(line) + " msg: Invalid file header");
		}
		++line;

		// skip comments
		if (cmd.front() == '#') {
			is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			continue;
		}

		// valid header commands start with '.' or end the header with BEGIN
		if (cmd.front() != '.' && cmd != "BEGIN" && cmd != "begin") {
			throw QFRException("[tfc parser] l:" + std::to_string(line) + " msg: Invalid file header");
		}

		// header read complete
		if (cmd == "BEGIN" || cmd == "begin")
			break;

		if (cmd == ".v") {
			is >> std::ws;
			std::getline(is, identifier);
			while ((pos = identifier.find(delimiter)) != std::string::npos) {
				variable = identifier.substr(0, pos);
				variables.emplace_back(variable);
				identifier.erase(0, pos+1);
			}
			variables.emplace_back(identifier);
		} else if (cmd == ".i") {
			is >> std::ws;
			std::getline(is, identifier);
			while ((pos = identifier.find(delimiter)) != std::string::npos) {
				variable = identifier.substr(0, pos);
				if (std::find(variables.begin(), variables.end(), variable) != variables.end()) {
					inputs.emplace_back(variable);
				} else {
					throw QFRException("[tfc parser] l:" + std::to_string(line) + " msg: Unknown variable in input statement: " + cmd);
				}
				identifier.erase(0, pos+1);
			}
			if (std::find(variables.begin(), variables.end(), identifier) != variables.end()) {
				inputs.emplace_back(identifier);
			} else {
				throw QFRException("[tfc parser] l:" + std::to_string(line) + " msg: Unknown variable in input statement: " + cmd);
			}
		} else if (cmd == ".o") {
			is >> std::ws;
			std::getline(is, identifier);
			while ((pos = identifier.find(delimiter)) != std::string::npos) {
				variable = identifier.substr(0, pos);
				if (std::find(variables.begin(), variables.end(), variable) != variables.end()) {
					outputs.emplace_back(variable);
				} else {
					throw QFRException("[tfc parser] l:" + std::to_string(line) + " msg: Unknown variable in output statement: " + cmd);
				}
				identifier.erase(0, pos+1);
			}
			if (std::find(variables.begin(), variables.end(), identifier) != variables.end()) {
				outputs.emplace_back(identifier);
			} else {
				throw QFRException("[tfc parser] l:" + std::to_string(line) + " msg: Unknown variable in output statement: " + cmd);
			}
		} else if (cmd == ".c") {
			is >> std::ws;
			std::getline(is, identifier);
			while ((pos = identifier.find(delimiter)) != std::string::npos) {
				variable = identifier.substr(0, pos);
				constants.emplace_back(variable);
				identifier.erase(0, pos+1);
			}
			constants.emplace_back(identifier);
		} else if (cmd == ".ol") { // ignore output labels
			is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			continue;
		} else {
			throw QFRException("[tfc parser] l:" + std::to_string(line) + " msg: Unknown command: " + cmd);
		}
	}
	addQubitRegister(inputs.size());
	auto nconstants = variables.size() - inputs.size();
	if (nconstants > 0) {
		addAncillaryRegister(nconstants);
	}

	auto qidx = 0;
	auto constidx = inputs.size();
	for (auto & var : variables) {
		// check if variable is input
		if (std::count(inputs.begin(), inputs.end(), var)) {
			varMap.insert({var, qidx++});
		} else {
			if (!constants.empty()) {
				if (constants.at(constidx-inputs.size()) == "0" || constants.at(constidx-inputs.size()) == "1") {
					// add X operation in case of initial value 1
					if (constants.at(constidx-inputs.size()) == "1")
						emplace_back<StandardOperation>(nqubits+nancillae, constidx, X);
					varMap.insert({var, constidx++});
				} else {
					throw QFRException("[tfc parser] l:" + std::to_string(line) + " msg: Non-binary constant specified: " + cmd);
				}
			} else {
				// variable does not occur in input statement --> assumed to be |0> ancillary
				varMap.insert({var, constidx++});
			}
		}
	}

	for (size_t q=0; q<variables.size(); ++q) {
		variable = variables.at(q);
		auto p = varMap.at(variable);
		initialLayout[q] = p;
		if (!outputs.empty()) {
			if (std::count(outputs.begin(), outputs.end(), variable)) {
				outputPermutation[q] = p;
			} else {
				outputPermutation.erase(q);
				garbage.set(p);
			}
		} else {
			// no output statement given --> assume all outputs are relevant
			outputPermutation[q] = p;
		}
	}

	return line;
}

void qc::QuantumComputation::readTFCGateDescriptions(std::istream& is, int line, std::map<std::string, unsigned short>& varMap) {
	std::regex gateRegex = std::regex("([tTfF])(\\d+)");
	std::smatch m;
	std::string cmd;

	while (!is.eof()) {
		if(!static_cast<bool>(is >> cmd)) {
			throw QFRException("[tfc parser] l:" + std::to_string(line) + " msg: Failed to read command");
		}
		++line;

		if (cmd.front() == '#') {
			is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			continue;
		}

		if (cmd == "END" || cmd == "end") break;
		else {
			// match gate declaration
			if (!std::regex_match(cmd, m, gateRegex)) {
				throw QFRException("[tfc parser] l:" + std::to_string(line) + " msg: Unsupported gate detected: " + cmd);
			}

			// extract gate information (identifier, #controls, divisor)
			OpType gate;
			if (m.str(1) == "t" || m.str(1) == "T") { // special treatment of t(offoli) for real format
				gate = X;
			} else {
				gate = SWAP;
			}
			unsigned short ncontrols = m.str(2).empty() ? 0 : static_cast<unsigned short>(std::stoul(m.str(2), nullptr, 0)) - 1;

			if (ncontrols >= nqubits+nancillae) {
				throw QFRException("[tfc parser] l:" + std::to_string(line) + " msg: Gate acts on " + std::to_string(ncontrols + 1) + " qubits, but only " + std::to_string(nqubits+nancillae) + " qubits are available.");
			}

			std::string qubits, label;
			is >> std::ws;
			getline(is, qubits);

			std::vector<Control> controls{ };

			std::string delimiter = ",";
			size_t pos = 0;

			while ((pos = qubits.find(delimiter)) != std::string::npos) {
				label = qubits.substr(0, pos);
				if (label.back() == '\'') {
					label.erase(label.size()-1);
					controls.emplace_back(varMap.at(label), Control::neg);
				} else {
					controls.emplace_back(varMap.at(label));
				}
				qubits.erase(0, pos+1);
			}
			controls.emplace_back(varMap.at(qubits));

			if (gate == X) {
				unsigned short target = controls.back().qubit;
				controls.pop_back();
				emplace_back<StandardOperation>(nqubits, controls, target);
			} else {
				unsigned short target0 = controls.back().qubit;
				controls.pop_back();
				unsigned short target1 = controls.back().qubit;
				controls.pop_back();
				emplace_back<StandardOperation>(nqubits, controls, target0, target1, gate);
			}
		}
	}
}
