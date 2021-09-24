/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Ecc.hpp"

//#include <stdlib.h>

Ecc::Ecc(struct EccInfo eccInfo, qc::QuantumComputation& quantumcomputation): ecc(eccInfo), qc(quantumcomputation) {
}

qc::QuantumComputation& Ecc::applyEcc() {
    qc.stripIdleQubits(true, false);
    statistics.nInputQubits = qc.getNqubits();
	statistics.nOutputQubits = qc.getNqubits()*ecc.nRedundantQubits;	//TODO remove if error case (no ECC) is handled correclty
	qcMapped.addQubitRegister(statistics.nOutputQubits);

	writeEccEncoding();

	long nInputGates = 0;
    for(auto& gate: qc) {
        nInputGates++;
        mapGate(gate);
    }
    statistics.nInputGates = nInputGates;

    writeEccDecoding();

	long nOutputGates = 0;
	for(auto& gate: qcMapped) {
        nOutputGates++;
    }
    statistics.nOutputGates = nOutputGates;
    statistics.success = true;

    return qcMapped;
}

std::ostream& Ecc::printResult(std::ostream& out) {
    out << "\tused error correcting code: " << ecc.name << std::endl;
    out << "\tgate overhead: " << statistics.getGateOverhead() << std::endl;
	out << "\tinput qubits: " << statistics.nInputQubits << std::endl;
	out << "\tinput gates: " << statistics.nInputGates << std::endl;
	out << "\toutput qubits: " << statistics.nOutputQubits << std::endl;
	out << "\toutput gates: " << statistics.nOutputGates << std::endl;
	return out;
}

void Ecc::dumpResult(const std::string& outputFilename) {
		if (qcMapped.empty()) {
			std::cerr << "Mapped circuit is empty." << std::endl;
			return;
		}

		size_t dot = outputFilename.find_last_of('.');
		std::string extension = outputFilename.substr(dot + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return ::tolower(c); });
		if (extension == "real") {
			dumpResult(outputFilename, qc::Real);
		} else if (extension == "qasm") {
			dumpResult(outputFilename, qc::OpenQASM);
		} else {
			throw qc::QFRException("[dump] Extension " + extension + " not recognized/supported for dumping.");
		}
	}

void Ecc::writeToffoli(unsigned short c1, unsigned short c2, unsigned short target) {
    const int nQubitsMapped = qcMapped.getNqubits();
    qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, target, qc::H);
    writeCnot(c2, target);
    qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, target, qc::Tdag);
    writeCnot(c1, target);
    qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, target, qc::T);
    writeCnot(c2, target);
    qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, target, qc::Tdag);
    writeCnot(c1, target);
    qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, target, qc::T);
    qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, target, qc::H);
    qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, c2, qc::Tdag);
    writeCnot(c1, c2);
    qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, c2, qc::Tdag);
    writeCnot(c1, c2);
    qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, c2, qc::S);
    qcMapped.emplace_back<qc::StandardOperation>(nQubitsMapped, c1, qc::T);
}

void Ecc::writeCnot(unsigned short control, unsigned short target) {
    dd::Control ctrl;
    ctrl.qubit = control;
    qcMapped.emplace_back<qc::StandardOperation>(qcMapped.getNqubits(), ctrl, target, qc::X);
}

