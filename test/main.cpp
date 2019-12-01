#include <functional>
#include <memory>

#include "StandardOperation.hpp"
#include "QuantumComputation.hpp"
#include "QFT.hpp"
#include "Grover.hpp"
#include "GoogleRandomCircuitSampling.hpp"

using namespace std;
using namespace chrono;

int main() {

	std::string filename = "./circuits/test.real";
	qc::Format format = qc::Real;
	qc::QuantumComputation qc;
	qc.import(filename, format);

	qc.reset();
	filename = "./circuits/test.qasm";
	format = qc::OpenQASM;
	qc.import(filename, format);
	qc.dump("test_dump.qasm", format);
	qc.import(filename, format);

	qc.reset();
	filename = "./circuits/grcs/bris_4_32_9_v2.txt";
	format = qc::GRCS;
	qc.import(filename, format);

	unsigned short n = 3;
	qc::QFT qft(n); // generates the QFT for n qubits

	n = 2;
	qc::Grover grover(n); // generates Grover's algorithm for a random n-bit oracle

	auto dd = make_unique<dd::Package>(); // create an instance of the DD package
	auto functionality = qft.buildFunctionality(dd);
	qft.printMatrix(dd, functionality);
	dd->export2Dot(functionality, "functionality.dot");


	auto initial_state = dd->makeZeroState(n+1); // create initial state |0...0>
	auto state_vector = grover.simulate(initial_state, dd);
	grover.printVector(dd, state_vector);
	dd->export2Dot(state_vector, "state_vector.dot", true);
	std::cout << grover << std::endl;

	return 0;
}
