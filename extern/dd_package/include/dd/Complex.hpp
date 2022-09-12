/*
 * This file is part of the MQT DD Package which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
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

        static Complex zero; // NOLINT(readability-identifier-naming,cppcoreguidelines-avoid-non-const-global-variables) automatic renaming does not work reliably, so skip linting
        static Complex one;  // NOLINT(readability-identifier-naming,cppcoreguidelines-avoid-non-const-global-variables) automatic renaming does not work reliably, so skip linting

        void setVal(const Complex& c) const {
            r->value = CTEntry::val(c.r);
            i->value = CTEntry::val(c.i);
        }

        [[nodiscard]] inline bool approximatelyEquals(const Complex& c) const {
            return CTEntry::approximatelyEquals(r, c.r) && CTEntry::approximatelyEquals(i, c.i);
        };

        [[nodiscard]] inline bool exactlyZero() const {
            return CTEntry::exactlyZero(r) && CTEntry::exactlyZero(i);
        };

        [[nodiscard]] inline bool exactlyOne() const {
            return CTEntry::exactlyOne(r) && CTEntry::exactlyZero(i);
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
        return os << c.toString();
    }

    inline Complex Complex::zero{&ComplexTable<>::zero, &ComplexTable<>::zero}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    inline Complex Complex::one{&ComplexTable<>::one, &ComplexTable<>::zero};   // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
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
