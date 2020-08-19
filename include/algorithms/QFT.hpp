/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QUANTUMFUNCTIONALITYBUILDER_QFT_H
#define QUANTUMFUNCTIONALITYBUILDER_QFT_H

#include "QuantumComputation.hpp"

namespace qc {
	class QFT : public QuantumComputation {

	public:
		explicit QFT(unsigned short nq);

		std::ostream& printStatistics(std::ostream& os) override;

		dd::Edge buildFunctionality(std::unique_ptr<dd::Package>& dd) override;

		dd::Edge simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) override;
	};
}

#endif //QUANTUMFUNCTIONALITYBUILDER_QFT_H
