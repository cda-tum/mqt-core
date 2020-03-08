#include "algorithms/BernsteinVazirani.hpp"

// BernsteinVazirani without Entanglement
namespace qc {
	/***
	 * Private Methods
	 ***/
	void BernsteinVazirani::setup(QuantumComputation& qc) {
		for (unsigned short i = 0; i < nqubits; ++i)
			emplace_back<StandardOperation>(nqubits, i, H);
	}

	void BernsteinVazirani::oracle(QuantumComputation& qc) {
		for (unsigned short i = 0; i < nqubits; ++i) {
			if (((hiddenInteger >> i) & 1) == 1) {
				emplace_back<StandardOperation>(nqubits, i, Z);
			}
		}
	}

	void BernsteinVazirani::postProcessing(QuantumComputation& qc) {
		for (unsigned short i = 0; i < nqubits; ++i)
			emplace_back<StandardOperation>(nqubits, i, H);
	}

	void BernsteinVazirani::full_BernsteinVazirani(QuantumComputation& qc) {
		
		// Generate circuit
		setup(qc);
		oracle(qc);
		postProcessing(qc);
	}

	/***
	 * Public Methods
	 ***/
	BernsteinVazirani::BernsteinVazirani(unsigned long hiddenInt) {
		
		// Save the hidden integer
		hiddenInteger = hiddenInt;
		
		// Determine the bitsize of the hidden integer
		while (hiddenInteger >> ++(size) > 0);

		// Prevents a ciruite with 0 qubits
		if (size == 0) {
			size = 1;
		}

		// Set nr of Qubits/ClassicalBits
		addQubitRegister(size);
		addClassicalRegister(size);
		
		// Circuite
		full_BernsteinVazirani(*this);
	}

	// Copy & Paste
	dd::Edge BernsteinVazirani::simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) {
		//TODO: Enhance this simulation routine // delegate to simulator
		return QuantumComputation::simulate(in, dd);
	}

	std::ostream& BernsteinVazirani::printStatistics(std::ostream& os) {
		os << "BernsteinVazirani (" << nqubits << ") Statistics:\n";
		os << "\tn: " << nqubits + 1 << std::endl;
		os << "\tm: " << getNindividualOps() << std::endl;
		os << "\tHiddenInteger: " << hiddenInteger << std::endl;
		os << "--------------" << std::endl;
		return os;
	}
}