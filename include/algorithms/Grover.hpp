/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QUANTUMFUNCTIONALITYBUILDER_GROVER_H
#define QUANTUMFUNCTIONALITYBUILDER_GROVER_H

#include <random>
#include <bitset>
#include <functional>

#include <QuantumComputation.hpp>

namespace qc {
	class Grover : public QuantumComputation {
	protected:
		std::function<unsigned long long()> oracleGenerator;

		void setup(QuantumComputation& qc);

		void oracle(QuantumComputation& qc);

		void diffusion(QuantumComputation& qc);

		void full_grover(QuantumComputation& qc);

		std::array<short, MAX_QUBITS> line{};

	public:
		unsigned int       seed         = 0;
		unsigned long long x            = 0;
		unsigned long long iterations   = 1;

		explicit Grover(unsigned short nq, unsigned int seed = 0);

		dd::Edge buildFunctionality(std::unique_ptr<dd::Package>& dd) override;

		dd::Edge simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) override;

		std::ostream& printStatistics(std::ostream& os) override;

	};
}

#endif //QUANTUMFUNCTIONALITYBUILDER_GROVER_H
