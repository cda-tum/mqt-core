/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_COMPLEX_HPP
#define DD_PACKAGE_COMPLEX_HPP

#include "ComplexTable.hpp"
#include "ComplexValue.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace dd {
    struct Complex {
        ComplexTable<>::Entry* r;
        ComplexTable<>::Entry* i;

        static Complex zero;
        static Complex one;

        void setVal(const Complex& c) const {
            r->value = c.r->val();
            i->value = c.i->val();
        }

        [[nodiscard]] bool approximatelyEquals(const Complex& c) const {
            return r->approximatelyEquals(c.r) && i->approximatelyEquals(c.i);
        };

        [[nodiscard]] bool approximatelyZero() const {
            return r->approximatelyZero() && i->approximatelyZero();
        }

        [[nodiscard]] bool approximatelyOne() const {
            return r->approximatelyOne() && i->approximatelyZero();
        }

        bool operator==(const Complex& other) const {
            return r == other.r && i == other.i;
        }

        bool operator!=(const Complex& other) const {
            return !operator==(other);
        }

        [[nodiscard]] std::string toString(bool formatted = true, int precision = -1) const {
            std::ostringstream ss{};

            if (precision >= 0) ss << std::setprecision(precision);

            auto real = r->val();
            auto imag = i->val();

            if (real != 0.) {
                if (formatted) {
                    ComplexValue::printFormatted(ss, real);
                } else {
                    ss << real;
                }
            }
            if (imag != 0.) {
                if (formatted) {
                    if (real == imag) {
                        ss << "(1+i)";
                        return ss.str();
                    } else if (imag == -real) {
                        ss << "(1-i)";
                        return ss.str();
                    }
                    ComplexValue::printFormatted(ss, imag, true);
                } else {
                    if (real == 0.) {
                        ss << imag;
                    } else {
                        if (imag > 0.) {
                            ss << "+";
                        }
                        ss << imag;
                    }
                    ss << "i";
                }
            }
            if (real == 0. && imag == 0.) return "0";

            return ss.str();
        }

        void writeBinary(std::ostream& os) const {
            r->writeBinary(os);
            i->writeBinary(os);
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const Complex& c) {
        auto r = c.r->val();
        auto i = c.i->val();

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
            auto h1 = std::hash<dd::ComplexTable<>::Entry*>{}(c.r);
            auto h2 = std::hash<dd::ComplexTable<>::Entry*>{}(c.i);
            return h1 ^ (h2 << 1);
        }
    };
} // namespace std

#endif //DD_PACKAGE_COMPLEX_HPP
