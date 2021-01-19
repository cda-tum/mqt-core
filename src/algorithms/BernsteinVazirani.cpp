#include "algorithms/BernsteinVazirani.hpp"

// BernsteinVazirani without Entanglement
namespace qc {
	/***
	 * Private Methods
	 ***/
	void BernsteinVazirani::setup() {
		for (unsigned short i = 0; i < nqubits; ++i)
			emplace_back<StandardOperation>(nqubits, i, H);
	}

	void BernsteinVazirani::oracle() {
		for (unsigned short i = 0; i < nqubits; ++i) {
			if (((hiddenInteger >> i) & 1) == 1) {
				emplace_back<StandardOperation>(nqubits, i, Z);
			}
		}
	}

	void BernsteinVazirani::postProcessing() {
		for (unsigned short i = 0; i < nqubits; ++i)
			emplace_back<StandardOperation>(nqubits, i, H);
	}

	void BernsteinVazirani::full_BernsteinVazirani() {
		// Generate circuit
		setup();
		oracle();
		postProcessing();
	}

	/***
	 * Public Methods
	 ***/
	BernsteinVazirani::BernsteinVazirani(unsigned long hiddenInteger) : hiddenInteger(hiddenInteger) {
        name = "bv_" + std::to_string(hiddenInteger);
		// Determine the bitsize of the hidden integer
		while (hiddenInteger >> ++(size) > 0);

		// Prevents a circuit with 0 qubits
		if (size == 0) {
			size = 1;
		}

		// Set nr of Qubits/ClassicalBits
		addQubitRegister(size);
		addClassicalRegister(size);
		
		// Circuit
		full_BernsteinVazirani();
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
