/*
 * This file is part of the MQT DD Package which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#ifndef DDpackage_NOISEOPERATIONTABLE_HPP
#define DDpackage_NOISEOPERATIONTABLE_HPP

#include "Definitions.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace dd {
    template<class Edge, std::size_t numberOfStochasticOperations = 64>
    class StochasticNoiseOperationTable {
    public:
        explicit StochasticNoiseOperationTable(const std::size_t nv):
            nvars(nv) { resize(nv); };

        // access functions
        [[nodiscard]] const auto& getTable() const { return table; }

        void resize(std::size_t nq) {
            nvars = nq;
            table.resize(nvars);
        }

        void insert(std::uint_fast8_t kind, Qubit target, const Edge& r) {
            assert(kind < numberOfStochasticOperations); // There are new operations in OpType. Increase the value of numberOfOperations accordingly
            table.at(static_cast<std::size_t>(target)).at(kind) = r;
            ++count;
        }

        Edge lookup(std::uint_fast8_t kind, Qubit target) {
            assert(kind < numberOfStochasticOperations); // There are new operations in OpType. Increase the value of numberOfOperations accordingly
            lookups++;
            Edge r{};
            auto entry = table.at(static_cast<std::size_t>(target)).at(kind);
            if (entry.p == nullptr) {
                return r;
            }
            hits++;
            return entry;
        }

        void clear() {
            if (count > 0) {
                for (auto& tableRow: table) {
                    for (auto& entry: tableRow) {
                        entry.p = nullptr;
                    }
                }
                count = 0;
            }
        }

        [[nodiscard]] fp hitRatio() const { return static_cast<fp>(hits) / static_cast<fp>(lookups); }

        std::ostream& printStatistics(std::ostream& os = std::cout) {
            os << "hits: " << hits << ", looks: " << lookups << ", ratio: " << hitRatio() << std::endl;
            return os;
        }

    private:
        std::size_t                                                 nvars;
        std::vector<std::array<Edge, numberOfStochasticOperations>> table;

        // operation table lookup statistics
        std::size_t hits    = 0;
        std::size_t lookups = 0;
        std::size_t count   = 0;
    };
} // namespace dd

#endif //DDpackage_NOISEOPERATIONTABLE_HPP
