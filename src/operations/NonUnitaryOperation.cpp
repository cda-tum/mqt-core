/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "operations/NonUnitaryOperation.hpp"

namespace qc {
    // Measurement constructor
	NonUnitaryOperation::NonUnitaryOperation(const unsigned short nq, const std::vector<unsigned short>& qubitRegister, const std::vector<unsigned short>& classicalRegister) 
		: NonUnitaryOperation(nq, classicalRegister, Measure) {
		//targets = qubitRegister;
		//targets = classicalRegister;
		assert(qubitRegister.size() == classicalRegister.size());
		// i-th qubit to be measured shall be measured into i-th classical register
		for (const auto& qubit: qubitRegister)
			controls.emplace_back(qubit);
	}
	NonUnitaryOperation::NonUnitaryOperation(unsigned short nq, unsigned short qubit, unsigned short clbit) {
		type = Measure;
		nqubits = nq;
		controls.emplace_back(qubit);
		targets.emplace_back(clbit);
		Operation::setName();
	}

	// Snapshot constructor
	NonUnitaryOperation::NonUnitaryOperation(const unsigned short nq, const std::vector<unsigned short>& qubitRegister, int n) 
		: NonUnitaryOperation(nq, qubitRegister, Snapshot) {
		parameter[0] = n;
	}	

	// General constructor
	NonUnitaryOperation::NonUnitaryOperation(const unsigned short nq, const std::vector<unsigned short>& qubitRegister, OpType op) {
		type = op;
		nqubits  = nq;
		targets  = qubitRegister;
		Operation::setName();
	}

    std::ostream& NonUnitaryOperation::print(std::ostream& os) const {
	    return print(os, standardPermutation);
	}

	std::ostream& NonUnitaryOperation::print(std::ostream& os, const std::map<unsigned short, unsigned short>& permutation) const {
		std::array<short, MAX_QUBITS> line{};
		line.fill(LINE_DEFAULT);

		switch (type) {
			case Measure:
				os << name << "\t";
				for (unsigned int q = 0; q < controls.size(); ++q) {
					line[permutation.at(controls[q].qubit)] = targets[q];
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
				os << name << "\t";
				for (const auto& t: targets) {
					if (permutation.find(t) != permutation.end())
						line[permutation.at(t)] = LINE_TARGET;
				}
				for (int i = 0; i < nqubits; ++i) {
					if (line[i] == LINE_TARGET) {
						os << "\033[31m" << "r\t" << "\033[0m";
					} else {
						os << "|\t";
					}
				}
				break;
			case Snapshot:
				os << name << "\t";
				for (const auto& t: targets) {
					if (permutation.find(t) != permutation.end())
						line[permutation.at(t)] = LINE_TARGET;
				}
				for (int i = 0; i < nqubits; ++i) {
					if (line[i] == LINE_TARGET) {
						os << "\033[33m" << "s\t" << "\033[0m";
					} else {
						os << "|\t";
					}
				}
				os << "\tp: (" << targets.size() << ") (" << parameter[1] << ")";
				break;
			case ShowProbabilities:
				os << name;
				break;
			case Barrier:
				os << name << "\t";
				for (const auto& t: targets) {
					if (permutation.find(t) != permutation.end())
						line[permutation.at(t)] = LINE_TARGET;
				}
				for (int i = 0; i < nqubits; ++i) {
					if (line[i] == LINE_TARGET) {
						os << "\033[32m" << "b\t" << "\033[0m";
					} else {
						os << "|\t";
					}
				}
				break;
			default:
				std::cerr << "Non-unitary operation with invalid type " << type << " detected. Proceed with caution!" << std::endl;
				break;
		}
		return os;
	}

	void NonUnitaryOperation::dumpOpenQASM(std::ostream& of, const regnames_t& qreg, const regnames_t& creg) const {
		switch (type) {
			case Measure: 
				if(isWholeQubitRegister(qreg, controls[0].qubit, controls.back().qubit) && 
				   isWholeQubitRegister(qreg, targets[0],        targets.back())) {
					of << "measure " << qreg[controls[0].qubit].first << " -> " << creg[targets[0]].first << ";" << std::endl;
				} else {
					for (unsigned int q = 0; q < controls.size(); ++q) {
						of << "measure " << qreg[controls[q].qubit].second << " -> " << creg[targets[q]].second << ";" << std::endl;
					}
				}
				break;
			case Reset: 
				if(isWholeQubitRegister(qreg, targets[0], targets.back())) {
					of << "reset " << qreg[targets[0]].first << ";" << std::endl;
				} else {
					for (const auto& target: targets) {
						of << "reset " << qreg[target].second << ";" << std::endl;
					}
				}
				break;
			case Snapshot: 
				if(!targets.empty()) {
					of << "snapshot(" << parameter[0] << ") ";
					
					for (unsigned int q = 0; q < targets.size(); ++q) {
						if(q > 0) {
							of << ", ";
						}
						of << qreg[targets[q]].second;
					}
					of << ";" << std::endl;
				}
				break;
			case ShowProbabilities: 
				of << "show_probabilities;" << std::endl;
				break;
			case Barrier: 
				if(isWholeQubitRegister(qreg, targets[0],        targets.back())) {
					of << "barrier " << qreg[targets[0]].first << ";" << std::endl;
				} else {
					for (const auto& target: targets) {
						of << "barrier " << qreg[target].second << ";" << std::endl;
					}
				}
				break;
			default:
				std::cerr << "Non-unitary operation with invalid type " << type << " detected. Proceed with caution!" << std::endl;
				break;
		}
	}

	void NonUnitaryOperation::dumpQiskit(std::ostream& of, const regnames_t& qreg, const regnames_t& creg, const char *) const {
		switch (type) {
			case Measure:
				if(isWholeQubitRegister(qreg, controls[0].qubit, controls.back().qubit) &&
				   isWholeQubitRegister(qreg, targets[0],        targets.back())) {
					of << "qc.measure(" << qreg[controls[0].qubit].first << ", " << creg[targets[0]].first << ")" << std::endl;
				} else {
					of << "qc.measure([";
					for (const auto& control : controls) {
						of << qreg[control.qubit].second << ", ";
					}
					of << "], [";
					for (unsigned short target : targets) {
						of << creg[target].second << ", ";
					}
					of << "])" << std::endl;
				}
				break;
			case Reset:
				if(isWholeQubitRegister(qreg, targets[0], targets.back())) {
					of << "append(Reset(), " << qreg[targets[0]].first << ", [])" << std::endl;
				} else {
					of << "append(Reset(), [";
					for (const auto& target: targets) {
						of << qreg[target].second << ", " << std::endl;
					}
					of << "], [])" << std::endl;
				}
				break;
			case Snapshot:
				if(!targets.empty()) {
					of << "qc.snapshot(" << parameter[0] << ", qubits=[";
					for (unsigned short target : targets) {
						of << qreg[target].second << ", ";
					}
					of << "])" << std::endl;
				}
				break;
			case ShowProbabilities:
				std::cerr << "No equivalent to show_probabilities statement in qiskit" << std::endl;
				break;
			case Barrier:
				if(isWholeQubitRegister(qreg, targets[0],        targets.back())) {
					of << "qc.barrier(" << qreg[targets[0]].first << ")" << std::endl;
				} else {
					of << "qc.barrier([";
					for (const auto& target: targets) {
						of  << qreg[target].first << ", ";
					}
					of << "])" << std::endl;
				}
				break;
			default:
				std::cerr << "Non-unitary operation with invalid type " << type << " detected. Proceed with caution!" << std::endl;
				break;
		}
	}

	dd::Edge NonUnitaryOperation::getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const {
		// these operations do not alter the current state
		if (type == ShowProbabilities || type == Barrier || type == Snapshot) {
			return dd->makeIdent(0, static_cast<short>(nqubits-1));
		}

		throw QFRException("DD for non-unitary operation not available!");
	}

	dd::Edge NonUnitaryOperation::getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& perm) const {
		// these operations do not alter the current state
		if (type == ShowProbabilities || type == Barrier || type == Snapshot) {
			return dd->makeIdent(0, static_cast<short>(nqubits-1));
		}

		throw QFRException("DD for non-unitary operation not available!");
	}

	dd::Edge NonUnitaryOperation::getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line ) const {
		// these operations do not alter the current state
		if (type == ShowProbabilities || type == Barrier || type == Snapshot) {
			return dd->makeIdent(0, static_cast<short>(nqubits-1));
		}

		throw QFRException("Non-unitary operation is not reversible! No inverse DD is available.");
	}

	bool NonUnitaryOperation::actsOn(unsigned short i) {
		if (type == Measure) {
			for (const auto c:controls) {
				if (c.qubit == i)
					return true;
			}
		} else if (type == Reset) {
			for (const auto t:targets) {
				if (t == i)
					return true;
			}
		}
		return false; // other non-unitary operations (e.g., barrier statements) may be ignored
	}
}
