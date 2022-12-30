/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include <QuantumComputation.hpp>
#include <chrono>

namespace qc {
    enum Layout { Rectangular,
                  Bristlecone };

    class GoogleRandomCircuitSampling: public QuantumComputation {
    public:
        std::vector<std::vector<std::unique_ptr<Operation>>> cycles{};
        Layout                                               layout     = Rectangular;
        std::string                                          pathPrefix = "../../../Benchmarks/GoogleRandomCircuitSampling/inst/";

        explicit GoogleRandomCircuitSampling(const std::string& filename);

        GoogleRandomCircuitSampling(std::string prefix, std::uint16_t device, std::uint16_t depth, std::uint16_t instance);

        GoogleRandomCircuitSampling(std::string prefix, std::uint16_t x, std::uint16_t y, std::uint16_t depth, std::uint16_t instance);

        void importGRCS(const std::string& filename);

        [[nodiscard, gnu::pure]] size_t getNops() const override;

        std::ostream& print(std::ostream& os) const override;

        std::ostream& printStatistics(std::ostream& os) const override;

        void removeCycles(const std::size_t ncycles) {
            if (ncycles > cycles.size() - 2) {
                std::stringstream ss{};
                ss << "Cannot remove " << ncycles << " cycles out of a circuit containing 1+" << cycles.size() - 2 << "+1 cycles.";
                throw QFRException(ss.str());
            }
            auto last = std::move(cycles.back());
            cycles.pop_back();
            for (std::size_t i = 0; i < ncycles; ++i) {
                cycles.pop_back();
            }
            cycles.emplace_back(std::move(last));
        }
    };
} // namespace qc
