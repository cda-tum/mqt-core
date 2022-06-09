/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DDpackage_NOISEOPERATIONTABLE_HPP
#define DDpackage_NOISEOPERATIONTABLE_HPP

#include "Definitions.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace dd {
    // noise operation kinds
    enum NoiseOperationKind : std::uint_fast8_t {
        none,
        I,
        X,
        Y,
        Z,
        ATrue,
        AFalse,
        opCount
    };

    template<class Edge>
    class NoiseOperationTable {
    public:
        explicit NoiseOperationTable(std::size_t nvars):
            nvars(nvars) { resize(nvars); };

        // access functions
        [[nodiscard]] const auto& getTable() const { return table; }

        void resize(std::size_t nq) {
            nvars = nq;
            table.resize(nvars);
        }

        void insert(NoiseOperationKind kind, Qubit target, const Edge& r) {
            table.at(target).at(kind) = r;
            ++count;
        }

        Edge lookup(QubitCount n, NoiseOperationKind kind, Qubit target) {
            lookups++;
            Edge r{};
            auto entry = table.at(target).at(kind);
            if (entry.p == nullptr) return r;
            if (entry.p->v != static_cast<Qubit>(n - 1)) return r;
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
            hits    = 0;
            lookups = 0;
        }

        [[nodiscard]] fp hitRatio() const { return static_cast<fp>(hits) / lookups; }
        std::ostream&    printStatistics(std::ostream& os = std::cout) {
            os << "hits: " << hits << ", looks: " << lookups << ", ratio: " << hitRatio() << std::endl;
            return os;
        }

    private:
        std::size_t                            nvars;
        static constexpr auto                  opCount = static_cast<std::uint_fast8_t>(NoiseOperationKind::opCount);
        std::vector<std::array<Edge, opCount>> table;

        // operation table lookup statistics
        std::size_t hits    = 0;
        std::size_t lookups = 0;
        std::size_t count   = 0;
    };
} // namespace dd

#endif //DDpackage_NOISEOPERATIONTABLE_HPP
