/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QCEC_RANDOMCLIFFORDCIRCUIT_HPP
#define QCEC_RANDOMCLIFFORDCIRCUIT_HPP

#include <random>
#include <functional>

#include <QuantumComputation.hpp>

namespace qc {
	class RandomCliffordCircuit : public QuantumComputation {
	protected:
		std::function<unsigned short()> cliffordGenerator;

		void append1QClifford(unsigned int idx, unsigned short target);
		void append2QClifford(unsigned int idx, unsigned short control, unsigned short target);

	public:
		unsigned int       depth        = 1;
		unsigned int       seed         = 0;

		explicit RandomCliffordCircuit(unsigned short nq, unsigned int depth = 1, unsigned int seed = 0);

		std::ostream& printStatistics(std::ostream& os) override;

	};
}

#endif //QCEC_RANDOMCLIFFORDCIRCUIT_HPP
