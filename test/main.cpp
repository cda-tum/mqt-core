/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"
#include "algorithms/Grover.hpp"
#include "algorithms/QFT.hpp"
#include "dd/Export.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"

using namespace std;

int main() {
    std::string            filename = "./circuits/test.real";
    qc::QuantumComputation qc(filename);

    filename = "./circuits/test.qasm";
    qc.import(filename);
    qc.dump("test_dump.qasm");

    filename = "./circuits/grcs/bris_4_40_9_v2.txt";
    qc.import(filename);

    dd::QubitCount n = 3;
    qc::QFT        qft(n); // generates the QFT for n qubits
    std::cout << qft << std::endl;

    n = 2;
    qc::Grover grover(n); // generates Grover's algorithm for a random n-bit oracle

    auto dd            = make_unique<dd::Package<>>(n + 1); // create an instance of the DD package
    auto functionality = buildFunctionality(&qft, dd);
    dd->printMatrix(functionality);
    dd::export2Dot(functionality, "functionality.dot");
    std::cout << std::endl;

    auto initial_state = dd->makeZeroState(n + 1); // create initial state |0...0>
    auto state_vector  = simulate(&grover, initial_state, dd);
    dd->printVector(state_vector);
    dd::export2Dot(state_vector, "state_vector.dot", true);
    std::cout << std::endl
              << grover << std::endl;

    return 0;
}
