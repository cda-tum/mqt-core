/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_CIRCUITOPTIMIZER_HPP
#define QFR_CIRCUITOPTIMIZER_HPP

#include "QuantumComputation.hpp"

#include <forward_list>
#include <map>

namespace qc {
    using DAG          = std::vector<std::forward_list<std::unique_ptr<Operation>*>>;
    using DAGIterator  = std::forward_list<std::unique_ptr<Operation>*>::iterator;
    using DAGIterators = std::vector<DAGIterator>;

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

        static void removeDiagonalGatesBeforeMeasureRecursive(DAG& dag, DAGIterators& dagIterators, unsigned short idx, const DAGIterator& until);

        static bool removeDiagonalGate(DAG& dag, DAGIterators& dagIterators, unsigned short idx, DAGIterator& it, qc::Operation* op);

        static void removeMarkedMeasurments(QuantumComputation& qc);

        static void removeFinalMeasurements(QuantumComputation& qc);

        static void removeFinalMeasurementsRecursive(DAG& dag, DAGIterators& DAGIterators, unsigned short idx, const DAGIterator& until);

        static bool removeFinalMeasurement(DAG& dag, DAGIterators& dagIterators, unsigned short idx, DAGIterator& it, qc::Operation* op);

        static void decomposeSWAP(QuantumComputation& qc, bool isDirectedArchitecture);

        static void decomposeTeleport(QuantumComputation& qc);
    };
} // namespace qc
#endif //QFR_CIRCUITOPTIMIZER_HPP
