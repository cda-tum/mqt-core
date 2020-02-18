/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"
#include "algorithms/QFT.hpp"
#include "algorithms/Grover.hpp"
#include "algorithms/GoogleRandomCircuitSampling.hpp"

using namespace std;
using namespace chrono;

int main() {
	std::string filename = "./circuits/test.real";
	qc::QuantumComputation qc(filename);

	filename = "./circuits/test.qasm";
	qc.import(filename);
	qc.dump("test_dump.qasm");

	filename = "./circuits/grcs/bris_4_40_9_v2.txt";
	qc.import(filename);

	unsigned short n = 3;
	qc::QFT qft(n); // generates the QFT for n qubits

	n = 2;
	qc::Grover grover(n); // generates Grover's algorithm for a random n-bit oracle

	auto dd = make_unique<dd::Package>(); // create an instance of the DD package
	auto functionality = qft.buildFunctionality(dd);
	qft.printMatrix(dd, functionality);
	dd->export2Dot(functionality, "functionality.dot");
	std::cout << std::endl;

	auto initial_state = dd->makeZeroState(n+1); // create initial state |0...0>
	auto state_vector = grover.simulate(initial_state, dd);
	grover.printVector(dd, state_vector);
	dd->export2Dot(state_vector, "state_vector.dot", true);
	std::cout << std::endl << grover << std::endl;
	
	return 0;
}
