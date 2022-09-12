/*
 * This file is part of the MQT DD Package which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_CONTROL_HPP
#define DD_PACKAGE_CONTROL_HPP

#include "Definitions.hpp"

#include <set>

namespace dd {
    struct Control {
        enum class Type : bool { pos = true,    // NOLINT(readability-identifier-naming)
                                 neg = false }; // NOLINT(readability-identifier-naming)

        Qubit qubit{};
        Type  type = Type::pos;
    };

    inline bool operator<(const Control& lhs, const Control& rhs) {
        return lhs.qubit < rhs.qubit || (lhs.qubit == rhs.qubit && lhs.type < rhs.type);
    }

    inline bool operator==(const Control& lhs, const Control& rhs) {
        return lhs.qubit == rhs.qubit && lhs.type == rhs.type;
    }

    inline bool operator!=(const Control& lhs, const Control& rhs) {
        return !(lhs == rhs);
    }

    // this allows a set of controls to be indexed by a `Qubit`
    struct CompareControl {
        using is_transparent [[maybe_unused]] = void;

        inline bool operator()(const Control& lhs, const Control& rhs) const {
            return lhs < rhs;
        }

        inline bool operator()(Qubit lhs, const Control& rhs) const {
            return lhs < rhs.qubit;
        }

        inline bool operator()(const Control& lhs, Qubit rhs) const {
            return lhs.qubit < rhs;
        }
    };
    using Controls = std::set<Control, CompareControl>;

    inline namespace literals {
        // NOLINTNEXTLINE(google-runtime-int) User-defined literals require unsigned long long int
        inline Control operator""_pc(unsigned long long int q) {
            return {static_cast<Qubit>(q)};
        }
        // NOLINTNEXTLINE(google-runtime-int) User-defined literals require unsigned long long int
        inline Control operator""_nc(unsigned long long int q) {
            return {static_cast<Qubit>(q), Control::Type::neg};
        }
    } // namespace literals
} // namespace dd

#endif //DD_PACKAGE_CONTROL_HPP
