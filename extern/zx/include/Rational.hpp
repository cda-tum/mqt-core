#pragma once

#include <gmpxx.h>
#include <iostream>
#include <stdint.h>

namespace zx {

    /*
 * Representation of fractions as multiples of pi
 * Rationals can only have values in the half-open interval (-1,1],
 * corresponding to the interval (-pi, pi]
 */
    class PiRational {
        void normalize();

    public:
        mpz_class num, denom;

        PiRational():
            num(0), denom(1){};
        explicit PiRational(int64_t num, int64_t denom):
            num(num), denom(denom) {
            normalize();
        }
        explicit PiRational(mpz_class num, mpz_class denom):
            num(num), denom(denom) {
            normalize();
        }
        explicit PiRational(int64_t num):
            num(num), denom(1) { normalize(); }
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
        bool isInteger() const { return denom == 1; }
        bool isZero() const { return num == 0; }
    };

    inline PiRational operator-(const PiRational& rhs) {
        return PiRational(-rhs.num, rhs.denom);
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
        return lhs.num * rhs.denom < rhs.num * lhs.denom;
    }

    inline bool operator<(const PiRational& lhs, int64_t rhs) {
        return lhs.num < rhs * lhs.denom;
    }

    inline bool operator<(int64_t lhs, const PiRational& rhs) {
        return lhs * rhs.denom < rhs.num;
    }

    inline bool operator<=(const PiRational& lhs, const PiRational& rhs) {
        return lhs.num * rhs.denom <= rhs.num * lhs.denom;
    }

    inline bool operator<=(const PiRational& lhs, int64_t rhs) {
        return lhs.num <= rhs * lhs.denom;
    }

    inline bool operator<=(int64_t lhs, const PiRational& rhs) {
        return lhs * rhs.denom <= rhs.num;
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
        return lhs.num == rhs.num && lhs.denom == rhs.denom;
    }

    inline bool operator==(const PiRational& lhs, int64_t rhs) {
        return lhs.num == rhs && lhs.denom == 1;
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
        os << rhs.num << "/" << rhs.denom;
        return os;
    }

} // namespace zx
