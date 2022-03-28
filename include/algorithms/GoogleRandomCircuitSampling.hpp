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

        GoogleRandomCircuitSampling(const std::string& pathPrefix, unsigned short device, unsigned short depth, unsigned short instance);

        GoogleRandomCircuitSampling(const std::string& pathPrefix, unsigned short x, unsigned short y, unsigned short depth, unsigned short instance);

        void importGRCS(const std::string& filename);

        [[nodiscard]] size_t getNops() const override;

        std::ostream& print(std::ostream& os) const override;

        std::ostream& printStatistics(std::ostream& os) const override;

        void removeCycles(unsigned short ncycles) {
            if (ncycles > cycles.size() - 2) {
                std::stringstream ss{};
                ss << "Cannot remove " << ncycles << " cycles out of a circuit containing 1+" << cycles.size() - 2 << "+1 cycles.";
                throw QFRException(ss.str());
            }
            auto last = std::move(cycles.back());
            cycles.pop_back();
            for (int i = 0; i < ncycles; ++i) {
                cycles.pop_back();
            }
            cycles.emplace_back(std::move(last));
        }
    };
} // namespace qc
