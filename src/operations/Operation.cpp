/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "Operation.hpp"

namespace qc {
	void Operation::setLine(std::array<short, MAX_QUBITS>& line) const {
		for(auto target: targets) {
			line[target] = LINE_TARGET;
		}
		for(auto control: controls) {
			line[control.qubit] = control.type == Control::pos? LINE_CONTROL_POS: LINE_CONTROL_NEG;
		}
	}

	void Operation::resetLine(std::array<short, MAX_QUBITS>& line) const {
		for(auto target: targets) {
			line[target] = LINE_DEFAULT;
		}
		for(auto control: controls) {
			line[control.qubit] = LINE_DEFAULT;
		}
	}

    std::ostream& Operation::print(std::ostream& os) const {
	    const auto prec_before = std::cout.precision(20);

	    os << name << "\t";
		std::array<short, MAX_QUBITS> line{};
		line.fill(-1);
		setLine(line);
		
	    for (int i = 0; i < nqubits; i++) {
			if (line[i] < 0) {
				os << "|\t";
			} else if (line[i] == LINE_CONTROL_NEG) {
				os << "\033[31m" << "c\t" << "\033[0m";
			} else if (line[i] == LINE_CONTROL_POS) {
				os << "\033[32m" << "c\t" << "\033[0m";
			} else {
				os << "\033[1m\033[36m" << name[0] << name[1] << "\t\033[0m";
			}
		}

		bool isZero = true;
		for (size_t i = 0; i < MAX_PARAMETERS; ++i) {
			if (parameter[i] != 0.L)
				isZero = false;
		}
		if (!isZero) {
			os << "\tp: ";
			CN::printFormattedReal(os, parameter[0]);
			os << " ";
			for (size_t j = 1; j < MAX_PARAMETERS; ++j) {
				isZero = true;
				for (size_t i = j; i < MAX_PARAMETERS; ++i) {
					if (parameter[i] != 0.L)
						isZero = false;
				}
				if (isZero) break;
				CN::printFormattedReal(os, parameter[j]);
				os << " ";
			}
		}

		std::cout.precision(prec_before);

		return os;
	}
}
