/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "operations/Operation.hpp"

#include <array>
#include <memory>

namespace qc {
    static constexpr std::array<qc::OpType, 8> DIAGONAL_GATES = {qc::I, qc::Z, qc::S, qc::Sdag, qc::T, qc::Tdag, qc::Phase, qc::RZ};

    class CircuitOptimizer {
    protected:
        static void addToDag(DAG& dag, std::unique_ptr<Operation>* op);
        static void addNonStandardOperationToDag(DAG& dag, std::unique_ptr<Operation>* op);

    public:
        CircuitOptimizer() = default;

        static DAG  constructDAG(QuantumComputation& qc);
        static void printDAG(const DAG& dag);
        static void printDAG(const DAG& dag, const DAGIterators& iterators);

        static void swapReconstruction(QuantumComputation& qc);

        static void singleQubitGateFusion(QuantumComputation& qc);

        static void removeIdentities(QuantumComputation& qc);

        static void removeDiagonalGatesBeforeMeasure(QuantumComputation& qc);

        static void removeFinalMeasurements(QuantumComputation& qc);

        static void decomposeSWAP(QuantumComputation& qc, bool isDirectedArchitecture);

        static void decomposeTeleport(QuantumComputation& qc);

        static void eliminateResets(QuantumComputation& qc);

        static void deferMeasurements(QuantumComputation& qc);

        static bool isDynamicCircuit(QuantumComputation& qc);

        static void reorderOperations(QuantumComputation& qc);

        static void flattenOperations(QuantumComputation& qc);

        static void cancelCNOTs(QuantumComputation& qc);

    protected:
        static void removeDiagonalGatesBeforeMeasureRecursive(DAG& dag, DAGReverseIterators& dagIterators, Qubit idx, const DAGReverseIterator& until);
        static bool removeDiagonalGate(DAG& dag, DAGReverseIterators& dagIterators, Qubit idx, DAGReverseIterator& it, qc::Operation* op);

        static void removeFinalMeasurementsRecursive(DAG& dag, DAGReverseIterators& dagIterators, Qubit idx, const DAGReverseIterator& until);
        static bool removeFinalMeasurement(DAG& dag, DAGReverseIterators& dagIterators, Qubit idx, DAGReverseIterator& it, qc::Operation* op);

        static void changeTargets(Targets& targets, const std::map<Qubit, Qubit>& replacementMap);
        static void changeControls(Controls& controls, const std::map<Qubit, Qubit>& replacementMap);

        using Iterator = decltype(qc::QuantumComputation::ops.begin());
        static Iterator flattenCompoundOperation(std::vector<std::unique_ptr<Operation>>& ops, Iterator it);
    };
} // namespace qc
