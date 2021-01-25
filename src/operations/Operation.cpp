/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "operations/Operation.hpp"

namespace qc {
	std::map<unsigned short, unsigned short> Operation::standardPermutation = Operation::create_standard_permutation();

	void Operation::setName() {
		switch (type) {
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
				strcpy(name, "U   ");
				break;
			case U2:
				strcpy(name, "U2  ");
				break;
			case Phase:
				strcpy(name, "P   ");
				break;
			case SX:
				strcpy(name, "SX  ");
				break;
			case SXdag:
				strcpy(name, "SXdg");
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
			case Peres:
				strcpy(name, "Pres");
				break;
			case Peresdag:
				strcpy(name, "Prdg");
				break;
			case Compound:
				strcpy(name, "Comp");
				break;
			case Measure:
				strcpy(name, "Meas");
				break;
			case Reset:
				strcpy(name, "Rst ");
				break;
			case Snapshot:
				strcpy(name, "Snap");
				break;
			case ShowProbabilities:
				strcpy(name, "Show probabilities");
				break;
			case Barrier:
				strcpy(name, "Barr");
				break;
			case ClassicControlled:
				strcpy(name, "clc_");
				break;
			default:
				throw QFRException("This constructor shall not be called for gate type (index) " + std::to_string((int)type));
		}
	}

	void Operation::setLine(std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const {
		for(const auto& target: targets) {
			#if DEBUG_MODE_OPERATIONS
			std::cout << "target = " << target << ", perm[target] = " << permutation.at(target) << std::endl;
			#endif

			line[permutation.at(target)] = LINE_TARGET;
		}
		for(const auto& control: controls) {
			#if DEBUG_MODE_OPERATIONS
			std::cout << "control = " << control.qubit << ", perm[control] = " << permutation.at(control.qubit) << std::endl;
			#endif

			line[permutation.at(control.qubit)] = control.type == Control::pos? LINE_CONTROL_POS: LINE_CONTROL_NEG;
		}
	}

	void Operation::resetLine(std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const {
		for(const auto& target: targets) {
			line[permutation.at(target)] = LINE_DEFAULT;
		}
		for(const auto& control: controls) {
			line[permutation.at(control.qubit)] = LINE_DEFAULT;
		}
	}

    std::ostream& Operation::print(std::ostream& os) const {
		return print(os, standardPermutation);
	}

	std::ostream& Operation::print(std::ostream& os, const std::map<unsigned short, unsigned short>& permutation) const {
		const auto prec_before = std::cout.precision(20);

		os << std::setw(4) << name << "\t";
		std::array<short, MAX_QUBITS> line{};
		line.fill(-1);
		setLine(line);

		for (const auto& physical_qubit: permutation) {
			unsigned short physical_qubit_index = physical_qubit.first;
			if (line[physical_qubit_index] == LINE_DEFAULT) {
				os << "|\t";
			} else if (line[physical_qubit_index] == LINE_CONTROL_NEG) {
				os << "\033[31m" << "c\t" << "\033[0m";
			} else if (line[physical_qubit_index] == LINE_CONTROL_POS) {
				os << "\033[32m" << "c\t" << "\033[0m";
			} else {
				if (type == ClassicControlled) {
					os << "\033[1m\033[35m" << name[2] << name[3];
				} else {
					os << "\033[1m\033[36m" << name[0] << name[1];
				}
				os << "\t\033[0m";
			}
		}

		bool isZero = true;
		for (size_t i = 0; i < MAX_PARAMETERS; ++i) {
			if (parameter[i] != 0.L)
				isZero = false;
		}
		if (!isZero) {
			os << "\tp: (";
			CN::printFormattedReal(os, parameter[0]);
			os << ") ";
			for (size_t j = 1; j < MAX_PARAMETERS; ++j) {
				isZero = true;
				for (size_t i = j; i < MAX_PARAMETERS; ++i) {
					if (parameter[i] != 0.L)
						isZero = false;
				}
				if (isZero) break;
				os << "(";
				CN::printFormattedReal(os, parameter[j]);
				os << ") ";
			}
		}

		std::cout.precision(prec_before);

		return os;
	}

}
