#pragma once

#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "operations/Operation.hpp"

#include <array>
#include <memory>
#include <unordered_set>

namespace qc {
static constexpr std::array<qc::OpType, 10> DIAGONAL_GATES = {
    qc::Barrier, qc::I,   qc::Z, qc::S,  qc::Sdg,
    qc::T,       qc::Tdg, qc::P, qc::RZ, qc::RZZ};

class CircuitOptimizer {
protected:
  static void addToDag(DAG& dag, std::unique_ptr<Operation>* op);
  static void addNonStandardOperationToDag(DAG& dag,
                                           std::unique_ptr<Operation>* op);

public:
  CircuitOptimizer() = default;

  static DAG constructDAG(QuantumComputation& qc);
  static void printDAG(const DAG& dag);
  static void printDAG(const DAG& dag, const DAGIterators& iterators);

  static void swapReconstruction(QuantumComputation& qc);

  static void singleQubitGateFusion(QuantumComputation& qc);

  static void removeIdentities(QuantumComputation& qc);

  static void removeDiagonalGatesBeforeMeasure(QuantumComputation& qc);

  static void removeFinalMeasurements(QuantumComputation& qc);

  static void decomposeSWAP(QuantumComputation& qc,
                            bool isDirectedArchitecture);

  static void decomposeTeleport(QuantumComputation& qc);

  static void eliminateResets(QuantumComputation& qc);

  static void deferMeasurements(QuantumComputation& qc);

  static bool isDynamicCircuit(QuantumComputation& qc);

  static void reorderOperations(QuantumComputation& qc);

  static void flattenOperations(QuantumComputation& qc);

  static void cancelCNOTs(QuantumComputation& qc);

  /**
   * @brief Replaces all MCX gates with MCZ gates (and H gates surrounding the
   * target qubit) in the given circuit.
   * @param qc the quantum circuit
   */
  static void replaceMCXWithMCZ(qc::QuantumComputation& qc) {
    replaceMCXWithMCZ(qc.ops);
  }

  /**
   * @brief Backpropagates the output permutation through the circuit.
   * @details Starts at the end of the circuit with a potentially incomplete
   * output permutation and backpropagates it through the circuit until the
   * beginning of the circuit is reached. The tracked permutation is updated
   * with every SWAP gate and, eventually, the initial layout of the circuit is
   * set to the tracked permutation. Any unassigned qubit in the initial layout
   * is assigned to the first available position (favoring an identity mapping).
   * @param qc the quantum circuit
   */
  static void backpropagateOutputPermutation(QuantumComputation& qc);

protected:
  static void removeDiagonalGatesBeforeMeasureRecursive(
      DAG& dag, DAGReverseIterators& dagIterators, Qubit idx,
      const qc::Operation* until);
  static bool removeDiagonalGate(DAG& dag, DAGReverseIterators& dagIterators,
                                 Qubit idx, DAGReverseIterator& it,
                                 qc::Operation* op);

  static void
  removeFinalMeasurementsRecursive(DAG& dag, DAGReverseIterators& dagIterators,
                                   Qubit idx, const qc::Operation* until);
  static bool removeFinalMeasurement(DAG& dag,
                                     DAGReverseIterators& dagIterators,
                                     Qubit idx, DAGReverseIterator& it,
                                     qc::Operation* op);

  static void changeTargets(Targets& targets,
                            const std::map<Qubit, Qubit>& replacementMap);
  static void changeControls(Controls& controls,
                             const std::map<Qubit, Qubit>& replacementMap);

  using Iterator = decltype(qc::QuantumComputation::ops.begin());
  static Iterator
  flattenCompoundOperation(std::vector<std::unique_ptr<Operation>>& ops,
                           Iterator it);

  static void replaceMCXWithMCZ(std::vector<std::unique_ptr<Operation>>& ops);
  static void backpropagateOutputPermutation(
      std::vector<std::unique_ptr<Operation>>& ops, Permutation& permutation,
      std::unordered_set<Qubit>& missingLogicalQubits);
};
} // namespace qc
