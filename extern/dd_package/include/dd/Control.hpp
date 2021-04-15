/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_CONTROL_HPP
#define DD_PACKAGE_CONTROL_HPP

#include "Definitions.hpp"

namespace dd {
    struct Control {
        enum class Type : bool { pos = true,
                                 neg = false };

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

    inline namespace literals {
        inline Control operator""_pc(unsigned long long int q) { return {static_cast<Qubit>(q)}; }
        inline Control operator""_nc(unsigned long long int q) { return {static_cast<Qubit>(q), Control::Type::neg}; }
    } // namespace literals
} // namespace dd

#endif //DD_PACKAGE_CONTROL_HPP
