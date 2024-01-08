#include "dd/Verification.hpp"

#include "QuantumComputation.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

#include <stdexcept>
#include <string>

namespace dd {

const std::vector<std::vector<OpType>> PRE_GENERATED_CIRCUITS_SIZE_1_1{{}};

const std::vector<std::vector<OpType>> PRE_GENERATED_CIRCUITS_SIZE_1_2{{Z}};

const std::vector<std::vector<OpType>> PRE_GENERATED_CIRCUITS_SIZE_2_1{{}};

const std::vector<std::vector<OpType>> PRE_GENERATED_CIRCUITS_SIZE_2_2{{Z}};

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

void addRandomGate(QuantumComputation& circuit, size_t n, Qubit d,
                   Qubit randomControl1, Qubit randomControl2,
                   Qubit randomTarget, OpType randomStandardOperation,
                   bool decomposeMcx) {
  if (randomStandardOperation == opTypeFromString("mcx")) {
    if (d >= 3) {
      if (decomposeMcx) {
        addDecomposedCxxGate(circuit, randomControl1, randomControl2,
                             randomTarget);
      } else {
        const Controls controls{randomControl1, randomControl2};
        StandardOperation op{n, controls, randomTarget,
                             randomStandardOperation};
        circuit.emplace_back<StandardOperation>();
      }
    }

  } else if (isTwoQubitGate(randomStandardOperation)) {
    if (d >= 2) {
      circuit.emplace_back<StandardOperation>(n, randomControl1, randomTarget,
                                              randomStandardOperation);
    }
  } else {
    if (d >= 1) {
      circuit.emplace_back<StandardOperation>(n, randomTarget,
                                              randomStandardOperation);
    }
  }
}

void addPreGeneratedCircuits(QuantumComputation& circuit1,
                             QuantumComputation& circuit2, size_t n,
                             Qubit groupBeginIndex, Qubit groupSize) {

  const auto& circuits1 = groupSize == 1 ? PRE_GENERATED_CIRCUITS_SIZE_1_1
                                         : PRE_GENERATED_CIRCUITS_SIZE_2_1;
  const auto& circuits2 = groupSize == 1 ? PRE_GENERATED_CIRCUITS_SIZE_1_2
                                         : PRE_GENERATED_CIRCUITS_SIZE_2_2;
  auto nrCircuits = circuits1.size();
  auto randomIndex = static_cast<size_t>(rand()) % nrCircuits;
  auto x1 = circuits1[randomIndex];
  auto x2 = circuits2[randomIndex];
  for (auto gateType : x1) {
    if (isTwoQubitGate(gateType)) {
      circuit1.emplace_back<StandardOperation>(n, groupBeginIndex,
                                               groupBeginIndex + 1, gateType);
    }
    circuit1.emplace_back<StandardOperation>(n, groupBeginIndex, gateType);
  }
  for (auto gateType : x2) {
    if (isTwoQubitGate(gateType)) {
      circuit2.emplace_back<StandardOperation>(n, groupBeginIndex,
                                               groupBeginIndex + 1, gateType);
    }
    circuit2.emplace_back<StandardOperation>(n, groupBeginIndex, gateType);
  }
}

std::tuple<Qubit, Qubit, Qubit> threeDiffferentRandomNumbers(Qubit min,
                                                             Qubit max) {
  auto range = max - min;
  auto randomTarget = static_cast<Qubit>(rand() % range) + min;
  Qubit randomControl1{0};
  Qubit randomControl2{0};
  if (range > 1) {
    randomControl1 = static_cast<Qubit>(rand() % (range - 1)) + min;
    if (randomControl1 == randomTarget) {
      randomControl1 = static_cast<Qubit>(max + min - 1);
    }
    if (range > 2) {
      randomControl2 = static_cast<Qubit>(rand() % (range - 2)) + min;
      if (randomControl2 == randomTarget) {
        randomControl2 = static_cast<Qubit>(max + min - 1);
      }
      if (randomControl2 == randomControl1) {
        randomControl2 = static_cast<Qubit>(max + min - 2);
        if (randomControl2 == randomTarget) {
          randomControl2 = static_cast<Qubit>(max + min - 1);
        }
      }
    }
  }
  return std::make_tuple(randomTarget, randomControl1, randomControl2);
}

std::pair<qc::QuantumComputation, qc::QuantumComputation>
generateRandomBenchmark(size_t n, Qubit d, Qubit m) {
  if (d > n) {
    throw std::runtime_error("The number of data or measured qubits can't be "
                             "bigger than the total number of qubits. n = " +
                             std::to_string(n) + ";d = " + std::to_string(d) +
                             "; m = " + std::to_string(m));
  }
  qc::QuantumComputation circuit1{n};
  qc::QuantumComputation circuit2{n};
  // 1) H gates
  for (Qubit i = 0U; i < d; i++) {
    circuit1.h(i);
    circuit2.h(i);
  }

  circuit1.barrier(0);
  circuit1.barrier(1);
  circuit1.barrier(2);
  circuit2.barrier(0);
  circuit2.barrier(1);
  circuit2.barrier(2);

  // 2) Totally equivalent subcircuits
  //    generate a random subcircuit with d qubits and 3d gates to apply on both
  //    circuits, but all the Toffoli gates in C2 are decomposed

  for (Qubit i = 0U; i < 3 * d; i++) {
    auto [randomTarget, randomControl1, randomControl2] =
        threeDiffferentRandomNumbers(0, d);
    auto randomStandardOperation = static_cast<OpType>(rand() % Compound);
    addRandomGate(circuit1, n, d, randomControl1, randomControl2, randomTarget,
                  randomStandardOperation, false);
    addRandomGate(circuit2, n, d, randomControl1, randomControl2, randomTarget,
                  randomStandardOperation, true);
  }
  circuit1.barrier(0);
  circuit1.barrier(1);
  circuit1.barrier(2);
  circuit2.barrier(0);
  circuit2.barrier(1);
  circuit2.barrier(2);
  // 3) Partially equivalent subcircuits

  //    divide data qubits into groups of size 1 or 2
  Qubit groupBeginIndex = 0;
  while (groupBeginIndex < d) {
    Qubit groupSize = 1;
    if (groupBeginIndex < d - 1) {
      groupSize = static_cast<Qubit>(rand() % 2) + 1;
    }

    addPreGeneratedCircuits(circuit1, circuit2, n, groupBeginIndex, groupSize);

    groupBeginIndex += groupSize;
  }
  circuit1.barrier(0);
  circuit1.barrier(1);
  circuit1.barrier(2);
  circuit2.barrier(0);
  circuit2.barrier(1);
  circuit2.barrier(2);
  // 4) Arbitrary gates
  //    arbitrary gates are added to not measured qubits
  if (d > m) {
    for (Qubit i = 0U; i < d - m; i++) {
      auto [randomTarget, randomControl1, randomControl2] =
          threeDiffferentRandomNumbers(m, d);
      auto randomStandardOperation = static_cast<OpType>(rand() % Compound);
      addRandomGate(circuit1, n, d - m, randomControl1, randomControl2,
                    randomTarget, randomStandardOperation, false);
    }
    for (Qubit i = 0U; i < d - m; i++) {
      auto [randomTarget, randomControl1, randomControl2] =
          threeDiffferentRandomNumbers(m, d);
      auto randomStandardOperation = static_cast<OpType>(rand() % Compound);
      addRandomGate(circuit2, n, d - m, randomControl1, randomControl2,
                    randomTarget, randomStandardOperation, false);
    }
  }
  circuit1.barrier(0);
  circuit1.barrier(1);
  circuit1.barrier(2);
  circuit2.barrier(0);
  circuit2.barrier(1);
  circuit2.barrier(2);
  // 5) CNOT gates (if there are ancilla qubits)
  Qubit currentDataQubit = 0;
  for (Qubit currentAncillaQubit = d;
       currentAncillaQubit < static_cast<Qubit>(n); currentAncillaQubit++) {
    auto nextDataQubit = static_cast<Qubit>((currentDataQubit + 1) % d);
    circuit1.cx(currentAncillaQubit, currentDataQubit);
    circuit2.cx(currentAncillaQubit, nextDataQubit);
    currentDataQubit = nextDataQubit;
  }

  return std::make_pair(circuit1, circuit2);
}
} // namespace dd
