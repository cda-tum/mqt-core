#pragma once

#if defined(GMP)
#include "boost/multiprecision/gmp.hpp"
using Rational = boost::multiprecision::mpq_rational;
using BigInt = boost::multiprecision::mpz_int;
#else
#include "boost/multiprecision/cpp_int.hpp"
using Rational = boost::multiprecision::cpp_rational;
using BigInt = boost::multiprecision::cpp_int;
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <utility>

namespace zx {

/*
 * Representation of fractions as multiples of pi
 * Rationals can only have values in the half-open interval (-1,1],
 * corresponding to the interval (-pi, pi]
 */
class PiRational {
public:
  PiRational() = default;
  explicit PiRational(int64_t num, int64_t denom) : frac(num, denom) {
    modPi();
  }
  explicit PiRational(const BigInt& num, const BigInt& denom)
      : frac(num, denom) {
    modPi();
  }
  explicit PiRational(int64_t num) : frac(num, 1) { modPi(); }
  explicit PiRational(double val);

  PiRational& operator+=(const PiRational& rhs);
  PiRational& operator+=(int64_t rhs);

  PiRational& operator-=(const PiRational& rhs);
  PiRational& operator-=(int64_t rhs);

  PiRational& operator*=(const PiRational& rhs);
  PiRational& operator*=(int64_t rhs);

  PiRational& operator/=(const PiRational& rhs);
  PiRational& operator/=(int64_t rhs);

  [[nodiscard]] bool isInteger() const {
    return boost::multiprecision::denominator(frac) == 1;
  }
  [[nodiscard]] bool isZero() const {
    return boost::multiprecision::numerator(frac) == 0;
  }
  [[nodiscard]] BigInt getDenom() const {
    return boost::multiprecision::denominator(frac);
  }

  [[nodiscard]] BigInt getNum() const {
    return boost::multiprecision::numerator(frac);
  }

  [[nodiscard]] double toDouble() const;

  [[nodiscard]] double toDoubleDivPi() const {
    return frac.convert_to<double>();
  }

  [[nodiscard]] bool isClose(const double x, const double tolerance) const {
    return std::abs(toDouble() - x) < tolerance;
  }

  [[nodiscard]] bool isCloseDivPi(const double x,
                                  const double tolerance) const {
    return std::abs(toDoubleDivPi() - x) < tolerance;
  }

  explicit operator double() const { return this->toDouble(); }

private:
  Rational frac{};

  void modPi();

  void setNum(const BigInt& num) {
    boost::multiprecision::numerator(frac) = num;
  }

  void setDenom(const BigInt& denom) const {
    boost::multiprecision::denominator(frac) = denom;
  }
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
  return (lhs.getNum() * rhs.getDenom()) < (rhs.getNum() * lhs.getDenom());
}

inline bool operator<(const PiRational& lhs, const int64_t rhs) {
  return lhs.getNum() < (rhs * lhs.getDenom());
}

inline bool operator<(const int64_t lhs, const PiRational& rhs) {
  return (lhs * rhs.getDenom()) < rhs.getNum();
}

inline bool operator<=(const PiRational& lhs, const PiRational& rhs) {
  return (lhs.getNum() * rhs.getDenom()) <= (rhs.getNum() * lhs.getDenom());
}

inline bool operator<=(const PiRational& lhs, const int64_t rhs) {
  return lhs.getNum() <= (rhs * lhs.getDenom());
}

inline bool operator<=(const int64_t lhs, const PiRational& rhs) {
  return (lhs * rhs.getDenom()) <= rhs.getNum();
}

inline bool operator>(const PiRational& lhs, const PiRational& rhs) {
  return rhs < lhs;
}

inline bool operator>(const PiRational& lhs, const int64_t rhs) {
  return rhs < lhs;
}

inline bool operator>(const int64_t lhs, const PiRational& rhs) {
  return rhs < lhs;
}

inline bool operator>=(const PiRational& lhs, const PiRational& rhs) {
  return rhs <= lhs;
}

inline bool operator>=(const PiRational& lhs, const int64_t rhs) {
  return rhs <= lhs;
}

inline bool operator>=(const int64_t lhs, const PiRational& rhs) {
  return rhs <= lhs;
}

inline bool operator==(const PiRational& lhs, const PiRational& rhs) {
  return lhs.getNum() == rhs.getNum() && lhs.getDenom() == rhs.getDenom();
}

inline bool operator==(const PiRational& lhs, const int64_t rhs) {
  return lhs.getNum() == rhs && lhs.getDenom() == 1;
}

inline bool operator==(const int64_t lhs, const PiRational& rhs) {
  return rhs == lhs;
}

inline bool operator!=(const PiRational& lhs, const PiRational& rhs) {
  return !(lhs == rhs);
}

inline bool operator!=(const PiRational& lhs, const int64_t rhs) {
  return !(lhs == rhs);
}

inline bool operator!=(const int64_t lhs, const PiRational& rhs) {
  return !(lhs == rhs);
}

inline std::ostream& operator<<(std::ostream& os, const zx::PiRational& rhs) {
  os << rhs.getNum() << "/" << rhs.getDenom();
  return os;
}

} // namespace zx
