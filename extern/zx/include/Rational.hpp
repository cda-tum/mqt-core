#ifndef JKQZX_INCLUDE_RATIONAL_HPP_
#define JKQZX_INCLUDE_RATIONAL_HPP_

#include <stdint.h>

namespace zx {
class Rational {
  void normalize();

public:
  int32_t num, denom;

  Rational() : num(0), denom(1){};
  Rational(int32_t num, int32_t denom) : num(num), denom(denom){};
  Rational(int32_t num) : num(num), denom(1){};
  Rational(double val);

  Rational &operator+=(const Rational &rhs);
  Rational &operator+=(const int32_t rhs);

  Rational &operator-=(const Rational &rhs);
  Rational &operator-=(const int32_t rhs);

  Rational &operator*=(const Rational &rhs);
  Rational &operator*=(const int32_t rhs);

  Rational &operator/=(const Rational &rhs);
  Rational &operator/=(const int32_t rhs);

  float to_float() const { return (static_cast<float>(num)) / denom; }
};

inline Rational operator-(const Rational &rhs) {
  return Rational(-rhs.num, rhs.denom);
}
inline Rational operator+(Rational lhs, const Rational &rhs) {
  lhs += rhs;
  return lhs;
}
inline Rational operator+(Rational lhs, const int32_t rhs) {
  lhs += rhs;
  return lhs;
}
inline Rational operator+(const int32_t lhs, Rational rhs) {
  rhs += lhs;
  return lhs;
}

inline Rational operator-(Rational lhs, const Rational &rhs) {
  lhs -= rhs;
  return lhs;
}
inline Rational operator-(Rational lhs, const int32_t rhs) {
  lhs -= rhs;
  return lhs;
}
inline Rational operator-(const int32_t lhs, Rational rhs) {
  rhs -= lhs;
  return lhs;
}

inline Rational operator*(Rational lhs, const Rational &rhs) {
  lhs *= rhs;
  return lhs;
}
inline Rational operator*(Rational lhs, const int32_t rhs) {
  lhs *= rhs;
  return lhs;
}
inline Rational operator*(const int32_t lhs, Rational rhs) {
  rhs *= lhs;
  return lhs;
}

inline Rational operator/(Rational lhs, const Rational &rhs) {
  lhs /= rhs;
  return lhs;
}
inline Rational operator/(Rational lhs, const int32_t rhs) {
  lhs /= rhs;
  return lhs;
}
inline Rational operator/(const int32_t lhs, Rational rhs) {
  rhs /= lhs;
  return lhs;
}

inline bool operator<(const Rational &lhs, const Rational &rhs) {
  return lhs.num * rhs.denom < rhs.num * lhs.denom;
}

inline bool operator<(const Rational &lhs, int32_t rhs) {
  return lhs.num < rhs * lhs.denom;
}

inline bool operator<(int32_t lhs, const Rational &rhs) {
  return lhs * rhs.denom < rhs.num;
}

inline bool operator<=(const Rational &lhs, const Rational &rhs) {
  return lhs.num * rhs.denom <= rhs.num * lhs.denom;
}

inline bool operator<=(const Rational &lhs, int32_t rhs) {
  return lhs.num <= rhs * lhs.denom;
}

inline bool operator<=(int32_t lhs, const Rational &rhs) {
  return lhs * rhs.denom <= rhs.num;
}

inline bool operator>(const Rational &lhs, const Rational &rhs) {
  return rhs < lhs;
}

inline bool operator>(const Rational &lhs, int32_t rhs) { return rhs < lhs; }

inline bool operator>(int32_t lhs, const Rational &rhs) { return rhs < lhs; }

inline bool operator>=(const Rational &lhs, const Rational &rhs) {
  return rhs <= lhs;
}

inline bool operator>=(const Rational &lhs, int32_t rhs) { return rhs <= lhs; }

inline bool operator>=(int32_t lhs, const Rational &rhs) { return rhs <= lhs; }

} // namespace zx
#endif /* JKQZX_INCLUDE_RATIONAL_HPP_ */
