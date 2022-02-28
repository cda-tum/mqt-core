/*
 * This file is part of the MQT DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "dd/Export.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"

#include <iostream>
#include <memory>

using namespace dd::literals;

auto BellCicuit1(std::unique_ptr<dd::Package>& dd) {
    /***** define Hadamard gate acting on q0 *****/
    auto h_gate = dd->makeGateDD(dd::Hmat, 2, 0);

    /***** define cx gate with control q0 and target q1 *****/
    auto cx_gate = dd->makeGateDD(dd::Xmat, 2, 0_pc, 1);

    //Multiply matrices to get functionality of circuit
    return dd->multiply(cx_gate, h_gate);
}

auto BellCicuit2(std::unique_ptr<dd::Package>& dd) {
    /***** define Hadamard gate acting on q1 *****/
    auto h_gate_q1 = dd->makeGateDD(dd::Hmat, 2, 1);

    /***** define Hadamard gate acting on q0 *****/
    auto h_gate_q0 = dd->makeGateDD(dd::Hmat, 2, 0);

    /***** define cx gate with control q1 and target q0 *****/
    auto cx_gate = dd->makeGateDD(dd::Xmat, 2, 1_pc, 0);

    //Multiply matrices to get functionality of circuit
    return dd->multiply(dd->multiply(h_gate_q1, h_gate_q0), dd->multiply(cx_gate, h_gate_q1));
}

int main() {
    dd::Package::printInformation(); // uncomment to print various sizes of structs and arrays
    //Initialize package
    auto dd = std::make_unique<dd::Package>(4);

    // create Bell circuit 1
    auto bell_circuit1 = BellCicuit1(dd);

    // create Bell circuit 2
    auto bell_circuit2 = BellCicuit2(dd);

    /***** Equivalence checking *****/
    if (bell_circuit1 == bell_circuit2) {
        std::cout << "Circuits are equal!" << std::endl;
    } else {
        std::cout << "Circuits are not equal!" << std::endl;
    }

    /***** Simulation *****/
    //Generate vector in basis state |00>
    auto zero_state = dd->makeZeroState(2);

    //Simulate the bell_circuit with initial state |00>
    auto bell_state  = dd->multiply(bell_circuit1, zero_state);
    auto bell_state2 = dd->multiply(bell_circuit2, zero_state);

    //print result
    dd->printVector(bell_state);

    std::cout << "Bell states have a fidelity of " << dd->fidelity(bell_state, bell_state2) << "\n";
    std::cout << "Bell state and zero state have a fidelity of " << dd->fidelity(bell_state, zero_state) << "\n";

    /***** Custom gates *****/
    // define, e.g., Pauli-Z matrix
    dd::GateMatrix m;
    m[0] = {1., 0.};
    m[1] = {0., 0.};
    m[2] = {0., 0.};
    m[3] = {-1., 0.};

    auto my_z_gate = dd->makeGateDD(m, 1, 0);
    std::cout << "DD of my gate has size " << dd->size(my_z_gate) << std::endl;

    // compute (partial) traces
    auto partTrace = dd->partialTrace(dd->makeIdent(2), {true, true});
    auto fullTrace = dd->trace(dd->makeIdent(4));
    std::cout << "Identity function for 4 qubits has trace: " << fullTrace << std::endl;

    /***** print DDs as SVG file *****/
    dd::export2Dot(bell_circuit1, "bell_circuit1.dot", false);
    dd::export2Dot(bell_circuit2, "bell_circuit2.dot");
    dd::export2Dot(bell_state, "bell_state.dot", true);
    dd::export2Dot(partTrace, "partial_trace.dot");

    /***** print statistics *****/
    dd->statistics();
    dd->garbageCollect(true);
}
