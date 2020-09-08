/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */
#include "QuantumComputation.hpp"

void qc::QuantumComputation::importGRCS(std::istream& is) {
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
		} else if (identifier == "is") {
			ss >> control;
			ss >> target;
			emplace_back<StandardOperation>(nqubits, std::vector<qc::Control>{ }, control, target, iSWAP);
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
				throw QFRException("[grcs parser] unknown gate '" + identifier + "'");
			}
		}
	}

	for (unsigned short i = 0; i < nqubits; ++i) {
		initialLayout.insert({ i, i});
		outputPermutation.insert({ i, i});
	}
}
