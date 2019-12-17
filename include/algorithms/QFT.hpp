/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QUANTUMFUNCTIONALITYBUILDER_QFT_H
#define QUANTUMFUNCTIONALITYBUILDER_QFT_H

#include "QuantumComputation.hpp"

namespace qc {
	class QFT : public QuantumComputation {
		bool performSwaps = false;
	public:
		explicit QFT(unsigned short nq, bool performSwaps = false);

		std::ostream& printStatistics(std::ostream& os = std::cout) override;
	};
}

#endif //QUANTUMFUNCTIONALITYBUILDER_QFT_H
