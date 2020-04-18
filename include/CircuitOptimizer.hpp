/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QCEC_CIRCUITOPTIMIZER_HPP
#define QCEC_CIRCUITOPTIMIZER_HPP

#include "QuantumComputation.hpp"
#include <forward_list>

namespace qc {
	using DAG = std::vector<std::forward_list<Operation*>>;

	class CircuitOptimizer {

	protected:
		static void addToDag(DAG& dag, Operation* op);
	public:
		CircuitOptimizer() = default;

		static DAG constructDAG(QuantumComputation& qc);
		static void swapGateFusion(QuantumComputation& qc);

	};
}
#endif //QCEC_CIRCUITOPTIMIZER_HPP
