/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include <memory>
#include <iostream>
#include <vector>
#include <chrono>

#include <QuantumComputation.hpp>

#ifndef QUANTUMFUNCTIONALITYBUILDER_GRCS_H

namespace qc {
	enum Layout {Rectangular, Bristlecone};

	class GoogleRandomCircuitSampling : public QuantumComputation {
	public:
		std::vector<std::vector<std::unique_ptr<Operation>>> cycles{};
		Layout layout;
		std::string pathPrefix = "../../../Benchmarks/GoogleRandomCircuitSampling/inst/";

		explicit GoogleRandomCircuitSampling(const std::string& filename);

		GoogleRandomCircuitSampling(const std::string& pathPrefix, unsigned short device, unsigned short depth, unsigned short instance);

		GoogleRandomCircuitSampling(const std::string& pathPrefix, unsigned short x, unsigned short y, unsigned short depth, unsigned short instance);

		~GoogleRandomCircuitSampling() override = default;

		void importGRCS(const std::string& filename);

		size_t getNops() const override;

		std::ostream& print(std::ostream& os = std::cout) const override;

		std::ostream& printStatistics(std::ostream& os = std::cout) override;

		dd::Edge buildFunctionality(std::unique_ptr<dd::Package>& dd, bool applySwapToPermutation = false) override;

		dd::Edge simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd, bool applySwapToPermutation = false) override;

	};
}

#define QUANTUMFUNCTIONALITYBUILDER_GRCS_H

#endif //QUANTUMFUNCTIONALITYBUILDER_GRCS_H
