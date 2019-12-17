/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "NonUnitaryOperation.hpp"

namespace qc {
    // Measurement constructor
	NonUnitaryOperation::NonUnitaryOperation(const unsigned short nq, const std::vector<unsigned short>& qubitRegister, const std::vector<unsigned short>& classicalRegister) : op(Measure) {
		nqubits = nq;
		strcpy(name, "Measure");
		targets = qubitRegister;
		assert(qubitRegister.size() == classicalRegister.size());
		// i-th qubit to be measured shall be measured into i-th classical register
		for (const auto& qubit: qubitRegister)
			controls.emplace_back(qubit);
		targets = classicalRegister;
	}

	// Reset constructor
	NonUnitaryOperation::NonUnitaryOperation(const unsigned short nq, const std::vector<unsigned short>& qubitRegister) : op(Reset) {
		nqubits = nq;
		strcpy(name, "Reset");
		targets = qubitRegister;
	}

	// Snapshot constructor
	NonUnitaryOperation::NonUnitaryOperation(const unsigned short nq, const std::vector<unsigned short>& qubitRegister, int n) : op(Snapshot) {
		nqubits = nq;
		strcpy(name, "Snapshot");
		targets      = qubitRegister;
		parameter[0] = n;
	}

    std::ostream& NonUnitaryOperation::print(std::ostream& os) const {
	    std::array<short, MAX_QUBITS> line{};
	    line.fill(LINE_DEFAULT);

		switch (op) {
			case Measure: 
				os << "Meas\t";
				for (int q = 0; q < controls.size(); ++q) {
					line[controls[q].qubit] = targets[q];
				}
				for (int i = 0; i < nqubits; ++i) {
					if (line[i] >= 0) {
						os << "\033[34m" << line[i] << "\t" << "\033[0m";
					} else {
						os << "|\t";
					}
				}
				break;
			case Reset: 
				os << "Rst \t";
				setLine(line);
				for (int i = 0; i < nqubits; ++i) {
					if (line[i] == LINE_TARGET) {
						os << "\033[31m" << "r\t" << "\033[0m";
					} else {
						os << "|\t";
					}
				}
				break;
			case Snapshot:
				os << "Snap\t";
				setLine(line);
				for (int i = 0; i < nqubits; ++i) {
					if (line[i] == LINE_TARGET) {
						os << "\033[33m" << "s\t" << "\033[0m";
					} else {
						os << "|\t";
					}
				}
				os << "\tp: " << targets.size() << " " << parameter[1];
				break;
			case ShowProbabilities: os << "Show probabilities";
				break;
		}
		return os;
	}

	void NonUnitaryOperation::dumpOpenQASM(std::ofstream& of, const std::vector<std::string>& qreg, const std::vector<std::string>& creg) const {
		switch (op) { //TODO check if same register
			case Measure: 
				for (int q = 0; q < controls.size(); ++q) {
					of << "measure " << qreg[targets[q]] << " -> " << creg[controls[q].qubit] << std::endl;
				}
				break;
			case Reset: 
				for (auto target: targets) {
					of << "reset " << qreg[target] << std::endl;
				}
				break;
			case Snapshot: 
				if(targets.size() > 0) {
					of << "snapshot(" << parameter[0] << ") ";
					
					for (int q = 0; q < targets.size(); ++q) {
						if(q > 0) {
							of << ", ";
						}
						of << qreg[targets[q]];
					}
					of << ";" << std::endl;
				}
				break;
			case ShowProbabilities: 
				of << "show_probabilities" << std::endl;
				break;
		}
	}
}
