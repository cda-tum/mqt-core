//
// Created by Lukas Burgholzer on 06.11.19.
//

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
