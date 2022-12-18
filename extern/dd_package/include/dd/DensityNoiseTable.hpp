/*
 * This file is part of the MQT DD Package which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#ifndef DDpackage_NOISECOMPUTETABLE_HPP
#define DDpackage_NOISECOMPUTETABLE_HPP

#include "Definitions.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <utility>

namespace dd {

    /// Data structure for caching computed results of noise operations
    /// \tparam OperandType type of the operation's operand
    /// \tparam ResultType type of the operation's result
    /// \tparam NBUCKET number of hash buckets to use (has to be a power of two)
    template<class OperandType, class ResultType, std::size_t NBUCKET = 32768>
    class DensityNoiseTable { //todo Inherent from UnaryComputerTable
    public:
        DensityNoiseTable() = default;

        struct Entry {
            OperandType            operand;
            ResultType             result;
            std::vector<dd::Qubit> usedQubits;
        };

        static constexpr size_t MASK = NBUCKET - 1;

        // access functions
        [[nodiscard]] const auto& getTable() const { return table; }

        static std::size_t hash(const OperandType& a, const std::vector<Qubit>& usedQubits) {
            std::size_t i = 0;
            for (const auto qubit: usedQubits) {
                i = (i << 3U) + i * static_cast<std::size_t>(qubit) + static_cast<std::size_t>(qubit);
            }
            return (std::hash<OperandType>{}(a) + i) & MASK;
        }

        void insert(const OperandType& operand, const ResultType& result, const std::vector<Qubit>& usedQubits) {
            const auto key   = hash(operand, usedQubits);
            auto&      entry = table[key];
            entry.result     = result;
            entry.operand    = operand;
            entry.usedQubits = usedQubits;
            ++count;
        }

        ResultType lookup(const OperandType& operand, const std::vector<Qubit>& usedQubits) {
            ResultType result{};
            lookups++;
            const auto key   = hash(operand, usedQubits);
            auto&      entry = table[key];
            if (entry.result.p == nullptr) {
                return result;
            }
            if (entry.operand != operand) {
                return result;
            }
            if (entry.usedQubits != usedQubits) {
                return result;
            }
            hits++;
            return entry.result;
        }

        void clear() {
            if (count > 0) {
                for (auto& entry: table) {
                    entry.result.p = nullptr;
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
        std::array<Entry, NBUCKET> table{};
        // compute table lookup statistics
        std::size_t hits    = 0;
        std::size_t lookups = 0;
        std::size_t count   = 0;
    };
} // namespace dd

#endif //DDpackage_NOISECOMPUTETABLE_HPP
