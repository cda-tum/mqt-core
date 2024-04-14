#pragma once

#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "operations/Operation.hpp"

#include <array>
#include <cstddef>
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
  static void replaceMCXWithMCZ(qc::QuantumComputation& qc);

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

  /**
   * @brief Collects all operations in the circuit into blocks of a maximum
   * size.
   * @details The circuit is traversed and operations are collected into blocks
   * of a maximum size. The blocks are then appended to the circuit in the order
   * they were collected. Each block is realized as a compound operation.
   * Light optimizations are applied to the blocks, such as removing identity
   * gates and fusing single-qubit gates.
   * @param qc the quantum circuit
   * @param maxBlockSize the maximum size of a block
   */
  static void collectBlocks(QuantumComputation& qc, std::size_t maxBlockSize);
};
} // namespace qc
