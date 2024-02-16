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
    {}, {}, {}, {}, {X}, {X}};

const std::vector<std::vector<OpType>> PRE_GENERATED_CIRCUITS_SIZE_2_2{
    {Z}, {Tdg}, {S}, {Sdg}, {X, Z}, {Z, X}};

void addPreGeneratedCircuits(QuantumComputation& circuit1,
                             QuantumComputation& circuit2, const size_t n,
                             const qc::Qubit groupBeginIndex,
                             const qc::Qubit groupSize) {

  const auto& circuits1 = groupSize == 1 ? PRE_GENERATED_CIRCUITS_SIZE_1_1
                                         : PRE_GENERATED_CIRCUITS_SIZE_2_1;
  const auto& circuits2 = groupSize == 1 ? PRE_GENERATED_CIRCUITS_SIZE_1_2
                                         : PRE_GENERATED_CIRCUITS_SIZE_2_2;
  const auto nrCircuits = circuits1.size();
  auto randomGenerator = circuit1.getGenerator();
  std::uniform_int_distribution<size_t> randomDistribution(0, nrCircuits - 1);

  const auto randomIndex = randomDistribution(randomGenerator);
  const auto x1 = circuits1[randomIndex];
  const auto x2 = circuits2[randomIndex];
  for (auto gateType : x1) {
    if (gateType == X) { // add CNOT
      circuit1.emplace_back<StandardOperation>(n, groupBeginIndex,
                                               groupBeginIndex + 1, gateType);
    } else {
      circuit1.emplace_back<StandardOperation>(n, groupBeginIndex, gateType);
    }
  }
  for (auto gateType : x2) {
    if (gateType == X) { // add CNOT
      circuit2.emplace_back<StandardOperation>(n, groupBeginIndex,
                                               groupBeginIndex + 1, gateType);
    } else {
      circuit2.emplace_back<StandardOperation>(n, groupBeginIndex, gateType);
    }
  }
}

void addDecomposedCcxGate(QuantumComputation& circuit, const Controls& controls,
                          const qc::Qubit target) {
  const qc::Qubit control1 = controls.begin()->qubit;
  const qc::Qubit control2 = (++controls.begin())->qubit;
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

void addStandardOperationToCircuit(QuantumComputation& circuit,
                                   const StandardOperation& op,
                                   const bool decomposeCcx) {
  if (op.getType() == X && decomposeCcx && op.getControls().size() == 2) {
    // decompose toffoli gate
    addDecomposedCcxGate(circuit, op.getControls(), op.getTargets()[0]);
  } else {
    circuit.emplace_back<StandardOperation>(op);
  }
}

std::vector<qc::Qubit>
fiveDiffferentRandomNumbers(const qc::Qubit min, const qc::Qubit max,
                            std::mt19937_64 randomGenerator) {
  std::vector<qc::Qubit> numbers;

  for (qc::Qubit i = min; i < max; i++) {
    numbers.emplace_back(i);
  }
  std::shuffle(numbers.begin(), numbers.end(), randomGenerator);

  const int64_t lengthOutputVector{
      static_cast<int64_t>(std::min<size_t>(5UL, numbers.size()))};

  std::vector<qc::Qubit> outputVector(numbers.begin(),
                                      numbers.begin() + lengthOutputVector);
  return outputVector;
}

StandardOperation convertToStandardOperation(
    const size_t n, const size_t nrQubits, const OpType randomOpType,
    const qc::Qubit randomTarget1, const qc::Qubit randomTarget2,
    const fp randomParameter1, const fp randomParameter2,
    const fp randomParameter3, const Controls& randomControls) {

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

StandardOperation makeRandomStandardOperation(const size_t n,
                                              const qc::Qubit nrQubits,
                                              const qc::Qubit min,
                                              std::mt19937_64 randomGenerator) {
  const auto randomNumbers =
      fiveDiffferentRandomNumbers(min, min + nrQubits, randomGenerator);

  // choose one of the non-compound operations, but not "None", and also
  // not GPhase or I or Barrier
  std::uniform_int_distribution<> randomDistrOpType(H, RZX);
  auto randomOpType = static_cast<OpType>(randomDistrOpType(randomGenerator));
  const qc::Qubit randomTarget1 = randomNumbers[0];
  qc::Qubit randomTarget2{min};
  if (randomNumbers.size() > 1) {
    randomTarget2 = randomNumbers[1];
  };
  // choose random controls, but not more than available qubits
  std::uniform_int_distribution<size_t> randomDistrNrControls(0, 2);
  size_t nrControls = std::min(randomNumbers.size() - 3,
                               randomDistrNrControls(randomGenerator));
  if (randomNumbers.size() < 3) {
    nrControls = 0;
  }
  if (nrControls == 2) {
    // otherwise toffoli gates are almost never generated
    randomOpType = qc::X;
  }
  Controls randomControls{};
  for (size_t i = 0; i < nrControls; i++) {
    randomControls.emplace(randomNumbers[i + 2]);
  }
  std::uniform_real_distribution<fp> randomDistrParameters(0, 2 * PI);
  const fp randomParameter1 = randomDistrParameters(randomGenerator);
  const fp randomParameter2 = randomDistrParameters(randomGenerator);
  const fp randomParameter3 = randomDistrParameters(randomGenerator);
  // const std::vector<fp> randomParameters{PI, PI_2, PI_4};
  // const fp randomParameter1 =
  //     randomParameters[static_cast<size_t>(rand()) %
  //     randomParameters.size()];
  // const fp randomParameter2 =
  //     randomParameters[static_cast<size_t>(rand()) %
  //     randomParameters.size()];
  // const fp randomParameter3 =
  //     randomParameters[static_cast<size_t>(rand()) %
  //     randomParameters.size()];
  return convertToStandardOperation(
      n, nrQubits, randomOpType, randomTarget1, randomTarget2, randomParameter1,
      randomParameter2, randomParameter3, randomControls);
}

std::pair<qc::QuantumComputation, qc::QuantumComputation>
generateRandomBenchmark(const size_t n, const qc::Qubit d, const qc::Qubit m) {
  if (d > n || m > n) {
    throw std::runtime_error("The number of data or measured qubits can't be "
                             "bigger than the total number of qubits. n = " +
                             std::to_string(n) + "; d = " + std::to_string(d) +
                             "; m = " + std::to_string(m));
  }

  qc::QuantumComputation circuit1{n};
  qc::QuantumComputation circuit2{n};

  auto randomGenerator = circuit1.getGenerator();

  // 1) H gates
  for (qc::Qubit i = 0U; i < d; i++) {
    circuit1.h(i);
    circuit2.h(i);
  }

  circuit1.barrier();
  circuit2.barrier();

  // 2) Totally equivalent subcircuits
  // Generate a random subcircuit with d qubits and 3*d gates to apply
  // on both circuits, but all the Toffoli gates in circuit2 are decomposed
  for (qc::Qubit i = 0U; i < 3 * d; i++) {
    const auto op = makeRandomStandardOperation(n, d, 0, randomGenerator);
    addStandardOperationToCircuit(circuit1, op, false);
    addStandardOperationToCircuit(circuit2, op, true);
  }

  circuit1.barrier();
  circuit2.barrier();

  // 3) Partially equivalent subcircuits
  // Divide data qubits into groups of size 1 or 2. For each group, we apply
  // pre-generated subcircuits, which are pairwise partially equivalent.
  qc::Qubit groupBeginIndex = 0;
  std::uniform_int_distribution<qc::Qubit> randomDistrGroupSize(1, 2);
  while (groupBeginIndex < d) {
    qc::Qubit groupSize = 1;
    if (groupBeginIndex < d - 1) {
      groupSize = randomDistrGroupSize(randomGenerator);
    }

    addPreGeneratedCircuits(circuit1, circuit2, n, groupBeginIndex, groupSize);

    groupBeginIndex += groupSize;
  }

  circuit1.barrier();
  circuit2.barrier();

  // 4) Arbitrary gates
  // Arbitrary gates are added to data qubits that are not measured
  if (d > m) {
    const qc::Qubit notMQubits = d - m;
    for (qc::Qubit i = 0U; i < notMQubits; i++) {
      addStandardOperationToCircuit(
          circuit1,
          makeRandomStandardOperation(n, notMQubits, m, randomGenerator),
          false);
      addStandardOperationToCircuit(
          circuit2,
          makeRandomStandardOperation(n, notMQubits, m, randomGenerator),
          false);
    }
  }

  circuit1.barrier();
  circuit2.barrier();

  // 5) CNOT gates
  // For each ancilla qubit, add a CNOT that has the ancilla as control qubit,
  // and any data qubit as target. As ancilla qubits are initially set to 0 and
  // we haven't added any other gates to the ancilla qubits until now, these
  // CNOT gates do not affect the circuit.
  if (d > 0) {
    qc::Qubit currentDataQubit = 0;
    for (qc::Qubit currentAncillaQubit = d; currentAncillaQubit < n;
         currentAncillaQubit++) {
      auto nextDataQubit = (currentDataQubit + 1) % d;
      circuit1.cx(currentAncillaQubit, currentDataQubit);
      circuit2.cx(currentAncillaQubit, nextDataQubit);
      currentDataQubit = nextDataQubit;
    }
  }

  for (qc::Qubit i = d; i < n; i++) {
    circuit1.setLogicalQubitAncillary(i);
    circuit2.setLogicalQubitAncillary(i);
  }

  for (qc::Qubit i = m; i < n; i++) {
    circuit1.setLogicalQubitGarbage(i);
    circuit2.setLogicalQubitGarbage(i);
  }

  return std::make_pair(circuit1, circuit2);
}

} // namespace dd
