/*
 * This file is part of IIC-JKU QCEC library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include <iostream>
#include <string>
#include <locale>
#include <algorithm>
#include <random>
#include <functional>
#include <set>

#include "QuantumComputation.hpp"

void show_usage(const std::string& name) {
	std::cerr << "Usage: " << name << "<PATH_INPUT_FILE> <PATH_TO_OUTPUT_FILE> (--remove_gates X)" << std::endl;
	std::cerr << "Supported input file formats:" << std::endl;
	std::cerr << "  .real                       " << std::endl;
	std::cerr << "  .qasm                       " << std::endl;
	std::cerr << "Supported output file formats:" << std::endl;
	std::cerr << "  .qasm                       " << std::endl;
	std::cerr << "  .py (qiskit)                " << std::endl;
	std::cerr << "If '--remove_gates X' is specified, X gates are randomly removed" << std::endl;
}

int main(int argc, char** argv){
	if (argc != 3 && argc != 5) {
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
	std::transform(extension.begin(), extension.end(), extension.begin(),
	        [](const unsigned char c) { return ::tolower(c);}
	        );
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
	std::transform(extension.begin(), extension.end(), extension.begin(),
	        [](const unsigned char c) { return ::tolower(c);}
	        );
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

	if (argc > 3) {
		unsigned long long gates_to_remove = std::stoull(argv[4]);

		std::array<std::mt19937_64::result_type , std::mt19937_64::state_size> random_data{};
		std::random_device rd;
		std::generate(begin(random_data), end(random_data), [&](){return rd();});
		std::seed_seq seeds(begin(random_data), end(random_data));
		std::mt19937_64 mt(seeds);
		std::uniform_int_distribution<unsigned long long> distribution(0, qc.getNops()-1);
		std::function<unsigned long long()> rng = [&]() { return distribution(mt); };

		std::set<unsigned long long> already_removed{};

		for (unsigned long long j=0; j < gates_to_remove; ++j) {
			auto gate_to_remove = rng() % qc.getNops();
			while (already_removed.count(gate_to_remove)) {
				gate_to_remove = rng() % qc.getNops();
			}
			already_removed.insert(gate_to_remove);
			auto it = qc.begin();
			if (it == qc.end()) continue;
			std::advance(it, gate_to_remove);

			qc.erase(it);
		}
	}

	qc.dump(outfile, outformat);

	return 0;
}
