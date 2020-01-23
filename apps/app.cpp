/*
 * This file is part of IIC-JKU QCEC library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include <iostream>
#include <memory>
#include <string>
#include <cctype>
#include <algorithm>

#include "QuantumComputation.hpp"

void show_usage(const std::string& name) {
	std::cerr << "Usage: " << name << "<PATH_INPUT_FILE> <PATH_TO_OUTPUT_FILE>" << std::endl;
	std::cerr << "Supported input file formats:" << std::endl;
	std::cerr << "  .real                       " << std::endl;
	std::cerr << "  .qasm                       " << std::endl;
	std::cerr << "Supported output file formats:" << std::endl;
	std::cerr << "  .qasm                       " << std::endl;
	std::cerr << "  .py (qiskit)                " << std::endl;

}

int main(int argc, char** argv){
	if (argc != 3) {
		show_usage(argv[0]);
		return 1;
	}

	// get filenames
	std::string infile = argv[1];
	std::string outfile = argv[2];

	// get file format
	qc::Format informat;
	size_t dot = infile.find_last_of('.');
	std::string extension = infile.substr(dot + 1);
	std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
	if (extension == "real") {
		informat = qc::Real;
	} else if (extension == "qasm") {
		informat = qc::OpenQASM;
	} else {
		show_usage(argv[0]);
		return 1;
	}

	qc::Format outformat;
	dot = outfile.find_last_of('.');
	extension = outfile.substr(dot + 1);
	std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
	if (extension == "py") {
		outformat = qc::Qiskit;
	} else if (extension == "qasm") {
		outformat = qc::OpenQASM;
	} else {
		show_usage(argv[0]);
		return 1;
	}

	// read circuit
	qc::QuantumComputation qc;
	qc.import(infile, informat);

	qc.dump(outfile, outformat);

	return 0;
}
