#include "dd/Verification.hpp"

#include "QuantumComputation.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

#include <stdexcept>
#include <string>

namespace dd {

void addDecomposedCxxGate(QuantumComputation& circuit, Qubit control1,
                          Qubit control2, Qubit target) {
  circuit.h(target);
  circuit.cx(control1, target);
  circuit.tdg(target);
  circuit.cx(control2, target);
  circuit.t(target);
  circuit.cx(control1, target);
  circuit.t(control1);
  circuit.tdg(target);
  circuit.cx(control2, target);
  circuit.cx(control2, control1);
  circuit.t(target);
  circuit.t(control2);
  circuit.tdg(control1);
  circuit.h(target);
  circuit.cx(control2, control1);
}

std::pair<qc::QuantumComputation, qc::QuantumComputation>
generateRandomBenchmark(Qubit n, Qubit d, Qubit m) {
  if (d > n) {
    throw std::runtime_error("The number of data or measured qubits can't be "
                             "bigger than the total number of qubits. n = " +
                             std::to_string(n) + ";d = " + std::to_string(d) +
                             "; m = " + std::to_string(m));
  }
  qc::QuantumComputation circuit1{n};
  qc::QuantumComputation circuit2{n};
  // H gates
  for (Qubit i = 0U; i < d; i++) {
    circuit1.h(i);
    circuit2.h(i);
  }
  // Totally equivalent subcircuits
  // generate a random subcircuit with d qubits and 3d gates to apply on both
  // circuits, but all the Toffoli gates in C2 are decomposed

  for (Qubit i = 0U; i < 3 * d; i++) {
    auto randomTarget = static_cast<Qubit>(rand() % d);
    auto randomControl1 = static_cast<Qubit>(rand() % (d - 1));
    if (randomControl1 == randomTarget) {
      randomControl1 = d - 1;
    }
    auto randomControl2 = static_cast<Qubit>(rand() % (d - 2));
    if (randomControl2 == randomTarget) {
      randomControl2 = d - 1;
    }
    if (randomControl2 == randomControl1) {
      randomControl2 = d - 2;
    }
    auto randomStandardOperation = static_cast<OpType>(rand() % Compound);
    circuit1.emplace_back<StandardOperation>(n, randomTarget,
                                             randomStandardOperation);
    if (randomStandardOperation == opTypeFromString("mcx")) {
      addDecomposedCxxGate(circuit2, randomControl1, randomControl2,
                           randomTarget);
    } else {
      circuit2.emplace_back<StandardOperation>(n, randomTarget,
                                               randomStandardOperation);
    }
  }

  return std::make_pair(circuit1, circuit2);
}
} // namespace dd
