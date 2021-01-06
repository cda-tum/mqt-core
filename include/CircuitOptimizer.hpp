/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QCEC_CIRCUITOPTIMIZER_HPP
#define QCEC_CIRCUITOPTIMIZER_HPP

#include "QuantumComputation.hpp"
#include <forward_list>
#include <map>

namespace qc {
	using DAG = std::vector<std::forward_list<std::unique_ptr<Operation>*>>;
	using DAGIterator = std::forward_list<std::unique_ptr<Operation>*>::iterator;
	using DAGIterators = std::vector<DAGIterator>;

	static constexpr std::array<qc::OpType, 8> diagonalGates = { qc::I, qc::Z, qc::S, qc::Sdag, qc::T, qc::Tdag, qc::Phase, qc::RZ};

	class CircuitOptimizer {

	protected:
		static const std::map<qc::OpType, qc::OpType> inverseMap;

		static void addToDag(DAG& dag, std::unique_ptr<Operation> *op);

	public:
		CircuitOptimizer() = default;

		static DAG constructDAG(QuantumComputation& qc);

		static void swapGateFusion(QuantumComputation& qc);

		static void singleQubitGateFusion(QuantumComputation& qc);

		static void removeIdentities(QuantumComputation& qc);

		static void removeDiagonalGatesBeforeMeasure(QuantumComputation& qc);

		static void removeDiagonalGatesBeforeMeasureRecursive(DAG& dag, DAGIterators& dagIterators, unsigned short idx, const DAGIterator& until);

		static bool removeDiagonalGate(DAG& dag, DAGIterators& dagIterators, unsigned short idx, DAGIterator& it, qc::Operation* op);
	};
}
#endif //QCEC_CIRCUITOPTIMIZER_HPP
