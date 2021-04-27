/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_COMPLEX_HPP
#define DD_PACKAGE_COMPLEX_HPP

#include "ComplexTable.hpp"
#include "ComplexValue.hpp"

#include <cstddef>
#include <iostream>
#include <utility>

namespace dd {
    using CTEntry = ComplexTable<>::Entry;

    struct Complex {
        CTEntry* r;
        CTEntry* i;

        static Complex zero;
        static Complex one;

        void setVal(const Complex& c) const {
            r->value = CTEntry::val(c.r);
            i->value = CTEntry::val(c.i);
        }

        [[nodiscard]] inline bool approximatelyEquals(const Complex& c) const {
            return CTEntry::approximatelyEquals(r, c.r) && CTEntry::approximatelyEquals(i, c.i);
        };

        [[nodiscard]] inline bool approximatelyZero() const {
            return CTEntry::approximatelyZero(r) && CTEntry::approximatelyZero(i);
        }

        [[nodiscard]] inline bool approximatelyOne() const {
            return CTEntry::approximatelyOne(r) && CTEntry::approximatelyZero(i);
        }

        inline bool operator==(const Complex& other) const {
            return r == other.r && i == other.i;
        }

        inline bool operator!=(const Complex& other) const {
            return !operator==(other);
        }

        [[nodiscard]] std::string toString(bool formatted = true, int precision = -1) const {
            return ComplexValue::toString(CTEntry::val(r), CTEntry::val(i), formatted, precision);
        }

        void writeBinary(std::ostream& os) const {
            CTEntry::writeBinary(r, os);
            CTEntry::writeBinary(i, os);
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const Complex& c) {
        auto r = CTEntry::val(c.r);
        auto i = CTEntry::val(c.i);

        if (r != 0) {
            ComplexValue::printFormatted(os, r);
        }
        if (i != 0) {
            if (r == i) {
                os << "(1+i)";
                return os;
            } else if (i == -r) {
                os << "(1-i)";
                return os;
            }
            ComplexValue::printFormatted(os, i, true);
        }
        if (r == 0 && i == 0) os << 0;

        return os;
    }

    inline Complex Complex::zero{&ComplexTable<>::zero, &ComplexTable<>::zero};
    inline Complex Complex::one{&ComplexTable<>::one, &ComplexTable<>::zero};
} // namespace dd

namespace std {
    template<>
    struct hash<dd::Complex> {
        std::size_t operator()(dd::Complex const& c) const noexcept {
            auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(c.r));
            auto h2 = dd::murmur64(reinterpret_cast<std::size_t>(c.i));
            return dd::combineHash(h1, h2);
        }
    };
} // namespace std

#endif //DD_PACKAGE_COMPLEX_HPP
