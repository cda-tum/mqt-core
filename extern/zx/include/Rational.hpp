#ifndef JKQZX_INCLUDE_RATIONAL_HPP_
#define JKQZX_INCLUDE_RATIONAL_HPP_

#include <iostream>
#include <stdint.h>
#include <gmpxx.h>

namespace zx {
    
/*
 * Representation of fractions as multiples of pi
 * Rationals can only have values in the half-open interval (-1,1],
 * corresponding to the interval (-pi, pi]
 */
class PyRational {
  void normalize();
  
public:
  mpz_class num, denom;

  PyRational() : num(0), denom(1){};
  explicit PyRational(int64_t num, int64_t denom) : num(num), denom(denom) {
    normalize();
  }
  explicit PyRational(mpz_class num, mpz_class denom) : num(num), denom(denom) {
    normalize();
  }
  explicit PyRational(int64_t num) : num(num), denom(1) { normalize(); }
  explicit PyRational(double val);

  PyRational &operator+=(const PyRational &rhs);
  PyRational &operator+=(const int64_t rhs);

  PyRational &operator-=(const PyRational &rhs);
  PyRational &operator-=(const int64_t rhs);

  PyRational &operator*=(const PyRational &rhs);
  PyRational &operator*=(const int64_t rhs);

  PyRational &operator/=(const PyRational &rhs);
  PyRational &operator/=(const int64_t rhs);

  // double to_double() const;
  bool is_integer() const { return denom == 1; }
  bool is_zero() const {return num == 0;}
};

inline PyRational operator-(const PyRational &rhs) {
  return PyRational(-rhs.num, rhs.denom);
}
inline PyRational operator+(PyRational lhs, const PyRational &rhs) {
  lhs += rhs;
  return lhs;
}
inline PyRational operator+(PyRational lhs, const int64_t rhs) {
  lhs += rhs;
  return lhs;
}
inline PyRational operator+(const int64_t lhs, PyRational rhs) {
  rhs += lhs;
  return rhs;
}

inline PyRational operator-(PyRational lhs, const PyRational &rhs) {
  lhs -= rhs;
  return lhs;
}
inline PyRational operator-(PyRational lhs, const int64_t rhs) {
  lhs -= rhs;
  return lhs;
}
inline PyRational operator-(const int64_t lhs, PyRational rhs) {
  rhs -= lhs;
  return rhs;
}

inline PyRational operator*(PyRational lhs, const PyRational &rhs) {
  lhs *= rhs;
  return lhs;
}
inline PyRational operator*(PyRational lhs, const int64_t rhs) {
  lhs *= rhs;
  return lhs;
}
inline PyRational operator*(const int64_t lhs, PyRational rhs) {
  rhs *= lhs;
  return rhs;
}

inline PyRational operator/(PyRational lhs, const PyRational &rhs) {
  lhs /= rhs;
  return lhs;
}
inline PyRational operator/(PyRational lhs, const int64_t rhs) {
  lhs /= rhs;
  return lhs;
}
inline PyRational operator/(const int64_t lhs, PyRational rhs) {
  rhs /= lhs;
  return rhs;
}

inline bool operator<(const PyRational &lhs, const PyRational &rhs) {
  return lhs.num * rhs.denom < rhs.num * lhs.denom;
}

inline bool operator<(const PyRational &lhs, int64_t rhs) {
  return lhs.num < rhs * lhs.denom;
}

inline bool operator<(int64_t lhs, const PyRational &rhs) {
  return lhs * rhs.denom < rhs.num;
}

inline bool operator<=(const PyRational &lhs, const PyRational &rhs) {
  return lhs.num * rhs.denom <= rhs.num * lhs.denom;
}

inline bool operator<=(const PyRational &lhs, int64_t rhs) {
  return lhs.num <= rhs * lhs.denom;
}

inline bool operator<=(int64_t lhs, const PyRational &rhs) {
  return lhs * rhs.denom <= rhs.num;
}

inline bool operator>(const PyRational &lhs, const PyRational &rhs) {
  return rhs < lhs;
}

inline bool operator>(const PyRational &lhs, int64_t rhs) { return rhs < lhs; }

inline bool operator>(int64_t lhs, const PyRational &rhs) { return rhs < lhs; }

inline bool operator>=(const PyRational &lhs, const PyRational &rhs) {
  return rhs <= lhs;
}

inline bool operator>=(const PyRational &lhs, int64_t rhs) { return rhs <= lhs; }

inline bool operator>=(int64_t lhs, const PyRational &rhs) { return rhs <= lhs; }

inline bool operator==(const PyRational &lhs, const PyRational &rhs) {
  return lhs.num == rhs.num && lhs.denom == rhs.denom;
}

inline bool operator==(const PyRational &lhs, int64_t rhs) {
  return lhs.num == rhs && lhs.denom == 1;
}

inline bool operator==(int64_t lhs, const PyRational &rhs) { return rhs == lhs; }

inline bool operator!=(const PyRational &lhs, const PyRational &rhs) {
  return !(lhs == rhs);
}

inline bool operator!=(const PyRational &lhs, int64_t rhs) {
  return !(lhs == rhs);
}

inline bool operator!=(int64_t lhs, const PyRational &rhs) {
  return !(lhs == rhs);
}

inline std::ostream &operator<<(std::ostream &os, const zx::PyRational &rhs) {
  os << rhs.num << "/" << rhs.denom;
  return os;
}

} // namespace zx

#endif /* JKQZX_INCLUDE_RATIONAL_HPP_ */
