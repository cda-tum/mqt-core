/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_GRCS_H
#define QFR_GRCS_H

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

        MatrixDD buildFunctionality(std::unique_ptr<dd::Package>& dd) const override;
        MatrixDD buildFunctionality(std::unique_ptr<dd::Package>& dd, unsigned short ncycles) {
            if (ncycles < cycles.size() - 2) {
                removeCycles(cycles.size() - 2 - ncycles);
            }
            return buildFunctionality(dd);
        }
        VectorDD simulate(const VectorDD& in, std::unique_ptr<dd::Package>& dd) const override;
        VectorDD simulate(const VectorDD& in, std::unique_ptr<dd::Package>& dd, unsigned short ncycles) {
            if (ncycles < cycles.size() - 2) {
                removeCycles(cycles.size() - 2 - ncycles);
            }
            return simulate(in, dd);
        }

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

#endif //QFR_GRCS_H
