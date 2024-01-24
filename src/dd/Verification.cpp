#include "dd/Verification.hpp"

#include "QuantumComputation.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace dd {

const std::vector<std::vector<OpType>> PRE_GENERATED_CIRCUITS_SIZE_1_1{
    {}, {}, {}, {}};

const std::vector<std::vector<OpType>> PRE_GENERATED_CIRCUITS_SIZE_1_2{
    {Z}, {Tdg}, {S}, {Sdg}};

const std::vector<std::vector<OpType>> PRE_GENERATED_CIRCUITS_SIZE_2_1{
    {}, {}, {}, {}};

const std::vector<std::vector<OpType>> PRE_GENERATED_CIRCUITS_SIZE_2_2{
    {Z}, {Tdg}, {S}, {Sdg}};

void addDecomposedCxxGate(QuantumComputation& circuit, Qubit control1,
                          Qubit control2, Qubit target) {
  circuit.h(target);
  circuit.cx(control1, target);
  circuit.tdg(target);
  circuit.cx(control2, target);
  circuit.t(target);
  circuit.cx(control1, target);
  circuit.tdg(target);
  circuit.cx(control2, target);
  circuit.t(target);
  circuit.t(control1);
  circuit.h(target);
  circuit.cx(control2, control1);
  circuit.t(control2);
  circuit.tdg(control1);
  circuit.cx(control2, control1);
}

void addRandomStandardOperation(QuantumComputation& circuit,
                                StandardOperation op, bool decomposeMcx) {
  std::vector<Qubit> controls{};
  for (auto c : op.getControls()) { // they are at most 2
    controls.push_back(static_cast<Qubit>(c.qubit));
  }
  std::vector<Qubit> targets{};
  for (auto t : op.getTargets()) { // they are at most 2
    targets.push_back(static_cast<Qubit>(t));
  }

  if (op.getType() == qc::X && controls.size() == 2) {

    if (decomposeMcx) {
      addDecomposedCxxGate(circuit, controls[0], controls[1], targets[0]);
      return;
    }
  }

  circuit.emplace_back<StandardOperation>(op);
}

std::vector<Qubit> fiveDiffferentRandomNumbers(Qubit min, Qubit max) {
  std::vector<Qubit> numbers;

  for (Qubit i = min; i < max; i++) {
    numbers.push_back(i);
  }
  unsigned seed = 42;
  std::shuffle(numbers.begin(), numbers.end(),
               std::default_random_engine(seed));

  auto lengthOutputVector = std::min(5UL, numbers.size());
  std::vector<Qubit> outputVector(numbers.begin(),
                                  numbers.begin() +
                                      static_cast<int64_t>(lengthOutputVector));
  return outputVector;
}

StandardOperation convertToStandardOperation(
    size_t n, size_t nrQubits, OpType randomOpType, Qubit randomTarget1,
    Qubit randomTarget2, fp randomParameter1, fp randomParameter2,
    fp randomParameter3, const Controls& randomControls) {

  switch (randomOpType) {
    // two targets and zero parameters
  case qc::SWAP:
  case qc::iSWAP:
  case qc::iSWAPdg:
  case qc::Peres:
  case qc::Peresdg:
  case qc::DCX:
  case qc::ECR:
    if (nrQubits > 1) {
      return {n, randomControls, Targets{randomTarget1, randomTarget2},
              randomOpType};
    }
    break;
    // two targets and one parameter
  case qc::RXX:
  case qc::RYY:
  case qc::RZZ:
  case qc::RZX:
    if (nrQubits > 1) {
      return {n, randomControls, Targets{randomTarget1, randomTarget2},
              randomOpType, std::vector<fp>{randomParameter1}};
    }
    break;

    // two targets and two parameters
  case qc::XXminusYY:
  case qc::XXplusYY:
    if (nrQubits > 1) {
      return {n, randomControls, Targets{randomTarget1, randomTarget2},
              randomOpType,
              std::vector<fp>{randomParameter1, randomParameter2}};
    }
    break;

    // one target and zero parameters
  case qc::I:
  case qc::H:
  case qc::X:
  case qc::Y:
  case qc::Z:
  case qc::S:
  case qc::Sdg:
  case qc::T:
  case qc::Tdg:
  case qc::V:
  case qc::Vdg:
  case qc::SX:
  case qc::SXdg:
    return {n, randomControls, randomTarget1, randomOpType};
    // one target and three parameters
  case qc::U:
    return {
        n, randomControls, randomTarget1, randomOpType,
        std::vector<fp>{randomParameter1, randomParameter2, randomParameter3}};
    // one target and two parameters
  case qc::U2:
    return {n, randomControls, randomTarget1, randomOpType,
            std::vector<fp>{randomParameter1, randomParameter2}};
    // one target and one parameter
  case qc::P:
  case qc::RX:
  case qc::RY:
  case qc::RZ:
    return {n, randomControls, randomTarget1, randomOpType,
            std::vector<fp>{randomParameter1}};
  default:
    return {n, randomTarget1, qc::I};
  }
  return {n, randomTarget1, qc::I};
}

StandardOperation makeRandomStandardOperation(size_t n, Qubit nrQubits,
                                              Qubit min) {
  auto randomNumbers = fiveDiffferentRandomNumbers(min, min + nrQubits);
  // choose one of the non-compound operations, but not "None"
  auto randomOpType = static_cast<OpType>(rand() % (Vdg - H) + H);
  Qubit randomTarget1 = randomNumbers[0];
  Qubit randomTarget2{min};
  if (randomNumbers.size() > 1) {
    randomTarget2 = randomNumbers[1];
  };
  size_t nrControls =
      std::min(randomNumbers.size() - 3, static_cast<size_t>(rand() % 3));
  if (randomNumbers.size() < 3) {
    nrControls = 0;
  }
  if (nrControls == 2) {
    // otherwise Cnots are almost never generated
    randomOpType = qc::X;
  }
  Controls randomControls{};
  for (size_t i = 0; i < nrControls; i++) {
    randomControls.emplace(randomNumbers[i + 2]);
  }
  const std::vector<fp> randomParameters{PI, PI_2, PI_4};
  const fp randomParameter1 =
      randomParameters[static_cast<size_t>(rand()) % randomParameters.size()];
  const fp randomParameter2 =
      randomParameters[static_cast<size_t>(rand()) % randomParameters.size()];
  const fp randomParameter3 =
      randomParameters[static_cast<size_t>(rand()) % randomParameters.size()];
  return convertToStandardOperation(
      n, nrQubits, randomOpType, randomTarget1, randomTarget2, randomParameter1,
      randomParameter2, randomParameter3, randomControls);
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
  for (Qubit i = 0U; i < static_cast<Qubit>(n); i++) {
    circuit1.barrier(i);
    circuit2.barrier(i);
  }
  // 2) Totally equivalent subcircuits
  //    generate a random subcircuit with d qubits and 3d gates to apply
  //    on both circuits, but all the Toffoli gates in C2 are decomposed

  for (Qubit i = 0U; i < 3 * d; i++) {
    auto op = makeRandomStandardOperation(n, d, 0);
    addRandomStandardOperation(circuit1, op, false);
    addRandomStandardOperation(circuit2, op, true);
  }
  for (Qubit i = 0U; i < static_cast<Qubit>(n); i++) {
    circuit1.barrier(i);
    circuit2.barrier(i);
  }
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
  for (Qubit i = 0U; i < static_cast<Qubit>(n); i++) {
    circuit1.barrier(i);
    circuit2.barrier(i);
  }
  // 4) Arbitrary gates
  //    arbitrary gates are added to not measured qubits
  if (d > m) {
    Qubit notMQubits = d - m;
    for (Qubit i = 0U; i < notMQubits; i++) {
      auto op = makeRandomStandardOperation(n, notMQubits, m);
      addRandomStandardOperation(circuit1, op, false);
    }
    for (Qubit i = 0U; i < notMQubits; i++) {
      auto op = makeRandomStandardOperation(n, notMQubits, m);
      addRandomStandardOperation(circuit2, op, false);
    }
  }
  for (Qubit i = 0U; i < static_cast<Qubit>(n); i++) {
    circuit1.barrier(i);
    circuit2.barrier(i);
  }
  // 5) CNOT gates (if there are ancilla qubits)
  Qubit currentDataQubit = 0;
  for (Qubit currentAncillaQubit = d;
       currentAncillaQubit < static_cast<Qubit>(n); currentAncillaQubit++) {
    auto nextDataQubit = static_cast<Qubit>((currentDataQubit + 1) % d);
    circuit1.cx(currentAncillaQubit, currentDataQubit);
    circuit2.cx(currentAncillaQubit, nextDataQubit);
    currentDataQubit = nextDataQubit;
  }

  for (Qubit i = d; i < static_cast<Qubit>(n); i++) {
    circuit1.setLogicalQubitAncillary(i);
    circuit2.setLogicalQubitAncillary(i);
  }
  for (Qubit i = m; i < static_cast<Qubit>(n); i++) {
    circuit1.setLogicalQubitGarbage(i);
    circuit2.setLogicalQubitGarbage(i);
  }

  return std::make_pair(circuit1, circuit2);
}
} // namespace dd
