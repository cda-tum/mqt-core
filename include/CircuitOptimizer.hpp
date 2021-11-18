/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_CIRCUITOPTIMIZER_HPP
#define QFR_CIRCUITOPTIMIZER_HPP

#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "dd/Package.hpp"
#include "operations/Operation.hpp"

#include <array>
#include <memory>

namespace qc {
    static constexpr std::array<qc::OpType, 8> diagonalGates = {qc::I, qc::Z, qc::S, qc::Sdag, qc::T, qc::Tdag, qc::Phase, qc::RZ};

    class CircuitOptimizer {
    protected:
        static void addToDag(DAG& dag, std::unique_ptr<Operation>* op);

    public:
        CircuitOptimizer() = default;

        static DAG constructDAG(QuantumComputation& qc);

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

    protected:
        static void removeDiagonalGatesBeforeMeasureRecursive(DAG& dag, DAGReverseIterators& dagIterators, dd::Qubit idx, const DAGReverseIterator& until);
        static bool removeDiagonalGate(DAG& dag, DAGReverseIterators& dagIterators, dd::Qubit idx, DAGReverseIterator& it, qc::Operation* op);

        static void removeFinalMeasurementsRecursive(DAG& dag, DAGReverseIterators& dagIterators, dd::Qubit idx, const DAGReverseIterator& until);
        static bool removeFinalMeasurement(DAG& dag, DAGReverseIterators& dagIterators, dd::Qubit idx, DAGReverseIterator& it, qc::Operation* op);

        static void changeTargets(Targets& targets, const std::map<dd::Qubit, dd::Qubit>& replacementMap);
        static void changeControls(dd::Controls& controls, const std::map<dd::Qubit, dd::Qubit>& replacementMap);

        static void reorderOperationsRecursive(std::vector<std::unique_ptr<qc::Operation>>& ops, DAG& dag, DAGIterators& dagIterators, dd::Qubit idx, const DAGIterator& until);
    };
} // namespace qc
#endif //QFR_CIRCUITOPTIMIZER_HPP
