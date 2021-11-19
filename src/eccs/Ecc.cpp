/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Ecc.hpp"

Ecc::Ecc(struct Info eccInfo, qc::QuantumComputation& quantumcomputation, int measFreq, bool decomposeMC): ecc(eccInfo), qc(quantumcomputation), measureFrequency(measFreq), decomposeMultiControlledGates(decomposeMC) {
    decodingDone=false;
}

void Ecc::initMappedCircuit() {
    qc.stripIdleQubits(true, false);
    statistics.nInputQubits = qc.getNqubits();
    statistics.nInputClassicalBits = (int)qc.getNcbits();
	statistics.nOutputQubits = qc.getNqubits()*ecc.nRedundantQubits+ecc.nCorrectingBits;
	statistics.nOutputClassicalBits = statistics.nInputClassicalBits+ecc.nCorrectingBits;
	qcMapped.addQubitRegister(statistics.nOutputQubits);
	qcMapped.addClassicalRegister(statistics.nInputClassicalBits);
	qcMapped.addClassicalRegister(ecc.nCorrectingBits, "qecc");
}

qc::QuantumComputation& Ecc::apply() {
    initMappedCircuit();

	writeEncoding();

	long nInputGates = 0;
    for(const auto& gate: qc) {
        nInputGates++;
        mapGate(gate);
        if(measureFrequency>0 && nInputGates%measureFrequency==0) {
            measureAndCorrect();
        }
    }
    statistics.nInputGates = nInputGates;

    if(!decodingDone) {
        measureAndCorrect();
        writeDecoding();
        decodingDone = true;
    }


    statistics.nOutputGates = qcMapped.getNindividualOps();

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
		if (extension == "qasm") {
			dumpResult(outputFilename, qc::OpenQASM);
		} else {
			throw qc::QFRException("[dump] Extension " + extension + " not recognized/supported for dumping.");
		}
	}

void Ecc::gateNotAvailableError(const std::unique_ptr<qc::Operation> &gate) {
    throw qc::QFRException(std::string("Gate ") + gate.get()->getName() + " not possible to encode in error code " + ecc.name + "!");
}

void Ecc::writeToffoli(int target, int c1, bool p1, int c2, bool p2) {
    if(!p1) { qcMapped.x(c1);    }
    if(!p2) { qcMapped.x(c2);    }

    qcMapped.h(target);
    qcMapped.x(target, dd::Control{dd::Qubit(c2)});
    qcMapped.tdag(target);
    qcMapped.x(target, dd::Control{dd::Qubit(c1)});
    qcMapped.t(target);
    qcMapped.x(target, dd::Control{dd::Qubit(c2)});
    qcMapped.tdag(target);
    qcMapped.x(target, dd::Control{dd::Qubit(c1)});
    qcMapped.t(target);
    qcMapped.t(c2);
    qcMapped.h(target);
    qcMapped.x(c2, dd::Control{dd::Qubit(c1)});
    qcMapped.t(c1);
    qcMapped.tdag(c2);
    qcMapped.x(c2, dd::Control{dd::Qubit(c1)});

    if(!p1) { qcMapped.x(c1);    }
    if(!p2) { qcMapped.x(c2);    }
}

void Ecc::writeZToffoli(int target, int c1, bool p1, int c2, bool p2) {
    qcMapped.h(target);
    writeToffoli(target, c1, p1, c2, p2);
    qcMapped.h(target);
}

