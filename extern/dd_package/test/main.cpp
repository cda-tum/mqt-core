/*
 * This file is part of the MQT DD Package which is released under the MIT
 * license. See file README.md or go to
 * https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#include "dd/Export.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"

#include <iostream>
#include <memory>

using namespace dd::literals;

auto bellCicuit1(std::unique_ptr<dd::Package<>>& dd) {
  /***** define Hadamard gate acting on q0 *****/
  auto hGate = dd->makeGateDD(dd::Hmat, 2, 0);

  /***** define cx gate with control q0 and target q1 *****/
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 0_pc, 1);

  // Multiply matrices to get functionality of circuit
  return dd->multiply(cxGate, hGate);
}

auto bellCicuit2(std::unique_ptr<dd::Package<>>& dd) {
  /***** define Hadamard gate acting on q1 *****/
  auto hGateQ1 = dd->makeGateDD(dd::Hmat, 2, 1);

  /***** define Hadamard gate acting on q0 *****/
  auto hGateQ0 = dd->makeGateDD(dd::Hmat, 2, 0);

  /***** define cx gate with control q1 and target q0 *****/
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 1_pc, 0);

  // Multiply matrices to get functionality of circuit
  return dd->multiply(dd->multiply(hGateQ1, hGateQ0),
                      dd->multiply(cxGate, hGateQ1));
}

int main() {                         // NOLINT(bugprone-exception-escape)
  dd::Package<>::printInformation(); // uncomment to print various sizes of
                                     // structs and arrays
  // Initialize package
  auto dd = std::make_unique<dd::Package<>>(4);

  // create Bell circuit 1
  auto bellCircuit1 = bellCicuit1(dd);

  // create Bell circuit 2
  auto bellCircuit2 = bellCicuit2(dd);

  /***** Equivalence checking *****/
  if (bellCircuit1 == bellCircuit2) {
    std::cout << "Circuits are equal!" << std::endl;
  } else {
    std::cout << "Circuits are not equal!" << std::endl;
  }

  /***** Simulation *****/
  // Generate vector in basis state |00>
  auto zeroState = dd->makeZeroState(2);

  // Simulate the bell_circuit with initial state |00>
  auto bellState = dd->multiply(bellCircuit1, zeroState);
  auto bellState2 = dd->multiply(bellCircuit2, zeroState);

  // print result
  dd->printVector(bellState);

  std::cout << "Bell states have a fidelity of "
            << dd->fidelity(bellState, bellState2) << "\n";
  std::cout << "Bell state and zero state have a fidelity of "
            << dd->fidelity(bellState, zeroState) << "\n";

  /***** Custom gates *****/
  // define, e.g., Pauli-Z matrix
  dd::GateMatrix m;
  m[0] = {1., 0.};
  m[1] = {0., 0.};
  m[2] = {0., 0.};
  m[3] = {-1., 0.};

  auto myZGate = dd->makeGateDD(m, 1, 0);
  std::cout << "DD of my gate has size " << dd->size(myZGate) << std::endl;

  // compute (partial) traces
  auto partTrace = dd->partialTrace(dd->makeIdent(2), {true, true});
  auto fullTrace = dd->trace(dd->makeIdent(4));
  std::cout << "Identity function for 4 qubits has trace: " << fullTrace
            << std::endl;

  /***** print DDs as SVG file *****/
  dd::export2Dot(bellCircuit1, "bell_circuit1.dot", false);
  dd::export2Dot(bellCircuit2, "bell_circuit2.dot");
  dd::export2Dot(bellState, "bell_state.dot", true);
  dd::export2Dot(partTrace, "partial_trace.dot");

  /***** print statistics *****/
  dd->statistics();
  dd->garbageCollect(true);
}
