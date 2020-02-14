#include "DDpackage.h"
#include <iostream>

// X gate matrix
constexpr dd::Matrix2x2 Xmat = {{{ 0, 0 }, { 1, 0 } }, {{ 1, 0 }, { 0, 0 } }};
// Hadamard gate matrix
constexpr dd::Matrix2x2 Hmat = {{{ dd::SQRT_2, 0 }, { dd::SQRT_2,  0 }},
                                {{ dd::SQRT_2, 0 }, { -dd::SQRT_2, 0 }}};

dd::Edge BellCicuit1(dd::Package* dd) {
    /***** define Hadamard gate acting on q0 *****/

    // set control/target:
    //    -1 don't care
    //    0 neg. control
    //    1 pos. control
    //    2 target
    short line[2] = {2,-1};
    dd::Edge h_gate = dd->makeGateDD(Hmat, 2, line);

    /***** define cx gate with control q0 and target q1 *****/
    line[0] = 1;
    line[1] = 2;

    dd::Edge cx_gate = dd->makeGateDD(Xmat, 2, line);

    //Multiply matrices to get functionality of circuit
    return dd->multiply(cx_gate, h_gate);
}

dd::Edge BellCicuit2(dd::Package* dd) {
    /***** define Hadamard gate acting on q1 *****/
    short line[2] = {-1,2};
    dd::Edge h_gate_q1 = dd->makeGateDD(Hmat, 2, line);

	/***** define Hadamard gate acting on q0 *****/
	line[0] = 2;
    line[1] = -1;
    dd::Edge h_gate_q0 = dd->makeGateDD(Hmat, 2, line);

    /***** define cx gate with control q1 and target q0 *****/
    line[0] = 2;
    line[1] = 1;
    dd::Edge cx_gate = dd->makeGateDD(Xmat, 2, line);

    //Multiply matrices to get functionality of circuit
    return dd->multiply(dd->multiply(h_gate_q1, h_gate_q0), dd->multiply(cx_gate, h_gate_q1));
}

int main() {
    //dd::Package::printInformation(); // uncomment to print various sizes of structs and arrays
    //Initialize package
    auto* dd = new dd::Package;

    // create Bell circuit 1
    dd::Edge bell_circuit1 = BellCicuit1(dd);

    // create Bell circuit 2
    dd::Edge bell_circuit2 = BellCicuit2(dd);

    /***** Equivalence checking *****/
    if(dd::Package::equals(bell_circuit1, bell_circuit2)) {
        std::cout << "Circuits are equal!" << std::endl;
    } else {
        std::cout << "Circuits are not equal!" << std::endl;
    }

    /***** Simulation *****/
    //Generate vector in basis state |00>
    dd::Edge zero_state = dd->makeZeroState(2);

    //Simulate the bell_circuit with initial state |00>
    dd::Edge bell_state = dd->multiply(bell_circuit1, zero_state);
    dd::Edge bell_state2 = dd->multiply(bell_circuit2, zero_state);

    //print result
	dd->printVector(bell_state);

    std::cout << "Bell states have a fidelity of " << dd->fidelity(bell_state, bell_state2) << "\n";
    std::cout << "Bell state and zero state have a fidelity of " << dd->fidelity(bell_state, zero_state) << "\n";

    /***** Custom gates *****/
    // define, e.g., Pauli-Z matrix
    dd::Matrix2x2 m;
    m[0][0] = { 1, 0 };
    m[0][1] = { 0, 0 };
    m[1][0] = { 0, 0 };
    m[1][1] = { -1, 0 };

    short line[1] = {2}; // target on first line

    dd::Edge my_z_gate = dd->makeGateDD(m, 1, line);
	std::cout << "DD of my gate has size " << dd->size(my_z_gate) << std::endl;

	// compute (partial) traces
	dd::Edge partTrace = dd->partialTrace(dd->makeIdent(0, 1), std::bitset<dd::MAXN>(2));
	auto fullTrace = dd->trace(dd->makeIdent(0, 3));
	std::cout << "Identity function for 4 qubits has trace: " << fullTrace << std::endl;

    /***** print DDs as SVG file *****/
	dd->export2Dot(bell_circuit1, "bell_circuit1.dot", false);
	dd->export2Dot(bell_circuit2, "bell_circuit2.dot");
	dd->export2Dot(bell_state, "bell_state.dot", true);
	dd->export2Dot(partTrace, "partial_trace.dot");

	/***** print statistics *****/
	dd->statistics();
	dd->garbageCollect(true);

    delete dd;
}
