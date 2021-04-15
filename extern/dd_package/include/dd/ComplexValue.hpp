/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_COMPLEXVALUE_HPP
#define DD_PACKAGE_COMPLEXVALUE_HPP

#include "ComplexTable.hpp"
#include "Definitions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace dd {
    struct ComplexValue {
        fp r, i;

        [[nodiscard]] bool approximatelyEquals(const ComplexValue& c) const {
            return std::abs(r - c.r) < ComplexTable<>::tolerance() &&
                   std::abs(i - c.i) < ComplexTable<>::tolerance();
        }

        [[nodiscard]] bool approximatelyZero() const {
            return std::abs(r) < ComplexTable<>::tolerance() &&
                   std::abs(i) < ComplexTable<>::tolerance();
        }

        [[nodiscard]] bool approximatelyOne() const {
            return std::abs(r - 1.) < ComplexTable<>::tolerance() &&
                   std::abs(i) < ComplexTable<>::tolerance();
        }

        bool operator==(const ComplexValue& other) const {
            return r == other.r && i == other.i;
        }

        bool operator!=(const ComplexValue& other) const {
            return !operator==(other);
        }

        void readBinary(std::istream& is) {
            is.read(reinterpret_cast<char*>(&r), sizeof(decltype(r)));
            is.read(reinterpret_cast<char*>(&i), sizeof(decltype(i)));
        }

        void writeBinary(std::ostream& os) const {
            os.write(reinterpret_cast<const char*>(&r), sizeof(decltype(r)));
            os.write(reinterpret_cast<const char*>(&i), sizeof(decltype(i)));
        }

        void from_string(const std::string& real_str, std::string imag_str) {
            fp real = real_str.empty() ? 0. : std::stod(real_str);

            imag_str.erase(remove(imag_str.begin(), imag_str.end(), ' '), imag_str.end());
            imag_str.erase(remove(imag_str.begin(), imag_str.end(), 'i'), imag_str.end());
            if (imag_str == "+" || imag_str == "-") imag_str = imag_str + "1";
            fp imag = imag_str.empty() ? 0. : std::stod(imag_str);
            r       = {real};
            i       = {imag};
        }

        static void printFormatted(std::ostream& os, fp r, bool imaginary = false) {
            if (r == 0.L) {
                os << (std::signbit(r) ? "-" : "+") << "0" << (imaginary ? "i" : "");
                return;
            }
            auto n = std::log2(std::abs(r));
            auto m = std::log2(std::abs(r) / SQRT2_2);
            auto o = std::log2(std::abs(r) / PI);

            if (n == 0) { // +-1
                if (imaginary) {
                    os << (std::signbit(r) ? "-" : "+") << "i";
                } else
                    os << (std::signbit(r) ? "-" : "") << 1;
                return;
            }

            if (m == 0) { // +- 1/sqrt(2)
                if (imaginary) {
                    os << (std::signbit(r) ? "-" : "+") << u8"\u221a\u00bdi";
                } else {
                    os << (std::signbit(r) ? "-" : "") << u8"\u221a\u00bd";
                }
                return;
            }

            if (o == 0) { // +- pi
                if (imaginary) {
                    os << (std::signbit(r) ? "-" : "+") << u8"\u03c0i";
                } else {
                    os << (std::signbit(r) ? "-" : "") << u8"\u03c0";
                }
                return;
            }

            if (std::abs(n + 1) < ComplexTable<>::tolerance()) { // 1/2
                if (imaginary) {
                    os << (std::signbit(r) ? "-" : "+") << u8"\u00bdi";
                } else
                    os << (std::signbit(r) ? "-" : "") << u8"\u00bd";
                return;
            }

            if (std::abs(m + 1) < ComplexTable<>::tolerance()) { // 1/sqrt(2) 1/2
                if (imaginary) {
                    os << (std::signbit(r) ? "-" : "+") << u8"\u221a\u00bd \u00bdi";
                } else
                    os << (std::signbit(r) ? "-" : "") << u8"\u221a\u00bd \u00bd";
                return;
            }

            if (std::abs(o + 1) < ComplexTable<>::tolerance()) { // +-pi/2
                if (imaginary) {
                    os << (std::signbit(r) ? "-" : "+") << u8"\u00bd \u03c0i";
                } else
                    os << (std::signbit(r) ? "-" : "") << u8"\u00bd \u03c0";
                return;
            }

            if (std::abs(std::round(n) - n) < ComplexTable<>::tolerance() && n < 0) { // 1/2^n
                if (imaginary) {
                    os << (std::signbit(r) ? "-" : "+") << u8"\u00bd\u002a\u002a" << (int)std::round(-n) << "i";
                } else
                    os << (std::signbit(r) ? "-" : "") << u8"\u00bd\u002a\u002a" << (int)std::round(-n);
                return;
            }

            if (std::abs(std::round(m) - m) < ComplexTable<>::tolerance() && m < 0) { // 1/sqrt(2) 1/2^m
                if (imaginary) {
                    os << (std::signbit(r) ? "-" : "+") << u8"\u221a\u00bd \u00bd\u002a\u002a" << (int)std::round(-m) << "i";
                } else
                    os << (std::signbit(r) ? "-" : "") << u8"\u221a\u00bd \u00bd\u002a\u002a" << (int)std::round(-m);
                return;
            }

            if (std::abs(std::round(o) - o) < ComplexTable<>::tolerance() && o < 0) { // 1/2^o pi
                if (imaginary) {
                    os << (std::signbit(r) ? "-" : "+") << u8"\u00bd\u002a\u002a" << (int)std::round(-o) << u8" \u03c0i";
                } else
                    os << (std::signbit(r) ? "-" : "") << u8"\u00bd\u002a\u002a" << (int)std::round(-o) << u8" \u03c0";
                return;
            }

            if (imaginary) { // default
                os << (std::signbit(r) ? "" : "+") << r << "i";
            } else
                os << r;
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const ComplexValue& c) {
        if (c.r != 0) {
            ComplexValue::printFormatted(os, c.r);
        }
        if (c.i != 0) {
            if (c.r == c.i) {
                os << "(1+i)";
                return os;
            } else if (c.i == -c.r) {
                os << "(1-i)";
                return os;
            }
            ComplexValue::printFormatted(os, c.i, true);
        }
        if (c.r == 0 && c.i == 0) os << 0;

        return os;
    }
} // namespace dd

namespace std {
    template<>
    struct hash<dd::ComplexValue> {
        std::size_t operator()(dd::ComplexValue const& c) const noexcept {
            auto h1 = std::hash<dd::fp>{}(c.r);
            auto h2 = std::hash<dd::fp>{}(c.i);
            return h1 ^ (h2 << 1);
        }
    };
} // namespace std
#endif //DD_PACKAGE_COMPLEXVALUE_HPP
