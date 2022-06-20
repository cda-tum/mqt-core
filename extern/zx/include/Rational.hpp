#pragma once

#if defined(GMP)
    #include <gmpxx.h>
using Rational = mpq_class;
using BigInt   = mpz_class;
#else
    #include "boost/multiprecision/cpp_int.hpp"
using Rational = boost::multiprecision::cpp_rational;
using BigInt   = boost::multiprecision::cpp_int;
#endif

#include <iostream>
#include <stdint.h>

namespace zx {

    /*
 * Representation of fractions as multiples of pi
 * Rationals can only have values in the half-open interval (-1,1],
 * corresponding to the interval (-pi, pi]
 */
    class PiRational {
        // void normalize();
    public:
        PiRational():
            frac(){};
        explicit PiRational(int64_t num, int64_t denom):
            frac(num, denom) {
            modPi();
        }
        explicit PiRational(const BigInt& num, const BigInt& denom):
            frac(num, denom) {
            modPi();
        }
        explicit PiRational(int64_t num):
            frac(num, 1) {
            modPi();
        }
        explicit PiRational(double val);

        PiRational& operator+=(const PiRational& rhs);
        PiRational& operator+=(const int64_t rhs);

        PiRational& operator-=(const PiRational& rhs);
        PiRational& operator-=(const int64_t rhs);

        PiRational& operator*=(const PiRational& rhs);
        PiRational& operator*=(const int64_t rhs);

        PiRational& operator/=(const PiRational& rhs);
        PiRational& operator/=(const int64_t rhs);

        // double to_double() const;
        [[nodiscard]] bool isInteger() const {
#if defined(GMP)
            return frac.get_den() == 1;
#else
            return boost::multiprecision::denominator(frac) == 1;
#endif
        }
        bool isZero() const {
#if defined(GMP)
            return frac.get_num() == 0;
#else
            return boost::multiprecision::numerator(frac) == 0;
#endif
        }
        BigInt getDenom() const {
#if defined(GMP)
            return frac.get_den();
#else
            return boost::multiprecision::denominator(frac);
#endif
        }

        BigInt getNum() const {
#if defined(GMP)
            return frac.get_num();
#else
            return boost::multiprecision::numerator(frac);
#endif
        }

    private:
        Rational frac;

        void normalize() {
#if defined(GMP)
            frac.canonicalize();
#else
            // frac.normalize();
#endif
        }

        void modPi();

        void setNum(const BigInt& num) {
#if defined(GMP)
            frac.get_num() = num;
#else
            boost::multiprecision::numerator(frac)   = num;
#endif
        }

        void setDenom(const BigInt& denom) {
#if defined(GMP)
            frac.get_den() = denom;
#else
            boost::multiprecision::denominator(frac) = denom;
#endif
        }
#if defined(GMP)
        BigInt& getDenomUnsafe() {
            return frac.get_den();
            // #else
            //             return boost::multiprecision::denominator(frac);
        }
#endif

#if defined(GMP)
        BigInt& getNumUnsafe() {
            return frac.get_num();
            // #else
            //             return boost::multiprecision::numerator(frac);
        }
#endif
    };

    inline PiRational operator-(const PiRational& rhs) {
        return PiRational(-rhs.getNum(), rhs.getDenom());
    }
    inline PiRational operator+(PiRational lhs, const PiRational& rhs) {
        lhs += rhs;
        return lhs;
    }
    inline PiRational operator+(PiRational lhs, const int64_t rhs) {
        lhs += rhs;
        return lhs;
    }
    inline PiRational operator+(const int64_t lhs, PiRational rhs) {
        rhs += lhs;
        return rhs;
    }

    inline PiRational operator-(PiRational lhs, const PiRational& rhs) {
        lhs -= rhs;
        return lhs;
    }
    inline PiRational operator-(PiRational lhs, const int64_t rhs) {
        lhs -= rhs;
        return lhs;
    }
    inline PiRational operator-(const int64_t lhs, PiRational rhs) {
        rhs -= lhs;
        return rhs;
    }

    inline PiRational operator*(PiRational lhs, const PiRational& rhs) {
        lhs *= rhs;
        return lhs;
    }
    inline PiRational operator*(PiRational lhs, const int64_t rhs) {
        lhs *= rhs;
        return lhs;
    }
    inline PiRational operator*(const int64_t lhs, PiRational rhs) {
        rhs *= lhs;
        return rhs;
    }

    inline PiRational operator/(PiRational lhs, const PiRational& rhs) {
        lhs /= rhs;
        return lhs;
    }
    inline PiRational operator/(PiRational lhs, const int64_t rhs) {
        lhs /= rhs;
        return lhs;
    }
    inline PiRational operator/(const int64_t lhs, PiRational rhs) {
        rhs /= lhs;
        return rhs;
    }

    inline bool operator<(const PiRational& lhs, const PiRational& rhs) {
        return lhs.getNum() * rhs.getDenom() < rhs.getNum() * lhs.getDenom();
    }

    inline bool operator<(const PiRational& lhs, int64_t rhs) {
        return lhs.getNum() < rhs * lhs.getDenom();
    }

    inline bool operator<(int64_t lhs, const PiRational& rhs) {
        return lhs * rhs.getDenom() < rhs.getNum();
    }

    inline bool operator<=(const PiRational& lhs, const PiRational& rhs) {
        return lhs.getNum() * rhs.getDenom() <= rhs.getNum() * lhs.getDenom();
    }

    inline bool operator<=(const PiRational& lhs, int64_t rhs) {
        return lhs.getNum() <= rhs * lhs.getDenom();
    }

    inline bool operator<=(int64_t lhs, const PiRational& rhs) {
        return lhs * rhs.getDenom() <= rhs.getNum();
    }

    inline bool operator>(const PiRational& lhs, const PiRational& rhs) {
        return rhs < lhs;
    }

    inline bool operator>(const PiRational& lhs, int64_t rhs) {
        return rhs < lhs;
    }

    inline bool operator>(int64_t lhs, const PiRational& rhs) {
        return rhs < lhs;
    }

    inline bool operator>=(const PiRational& lhs, const PiRational& rhs) {
        return rhs <= lhs;
    }

    inline bool operator>=(const PiRational& lhs, int64_t rhs) {
        return rhs <= lhs;
    }

    inline bool operator>=(int64_t lhs, const PiRational& rhs) {
        return rhs <= lhs;
    }

    inline bool operator==(const PiRational& lhs, const PiRational& rhs) {
        return lhs.getNum() == rhs.getNum() && lhs.getDenom() == rhs.getDenom();
    }

    inline bool operator==(const PiRational& lhs, int64_t rhs) {
        return lhs.getNum() == rhs && lhs.getDenom() == 1;
    }

    inline bool operator==(int64_t lhs, const PiRational& rhs) {
        return rhs == lhs;
    }

    inline bool operator!=(const PiRational& lhs, const PiRational& rhs) {
        return !(lhs == rhs);
    }

    inline bool operator!=(const PiRational& lhs, int64_t rhs) {
        return !(lhs == rhs);
    }

    inline bool operator!=(int64_t lhs, const PiRational& rhs) {
        return !(lhs == rhs);
    }

    inline std::ostream& operator<<(std::ostream& os, const zx::PiRational& rhs) {
        os << rhs.getNum() << "/" << rhs.getDenom();
        return os;
    }

} // namespace zx
