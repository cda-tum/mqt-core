#include "Rational.hpp"
#include "Definitions.hpp"
#include "QuantumComputation.hpp" //TODO incorrect include
#include <cmath>

namespace zx {
long gcd(long a, long b) {
  long r;

  while (b != 0) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}

Rational::Rational(double val) : num(0), denom(1) {
  double mult_pi = PI / val;
  double nearest = std::round(mult_pi);
  if (std::abs(nearest - mult_pi) < qc::PARAMETER_TOLERANCE) {
    denom = static_cast<int>(nearest);
    num = 1;
    return;
  }

  val /= PI;

  double integral = std::floor(val);
  double frac = val - integral;

  long gcd_ = gcd(std::round(frac * MAX_DENOM), MAX_DENOM);

  denom = MAX_DENOM / gcd_;
  num = round(frac * MAX_DENOM) / gcd_;
}

void Rational::normalize() {
  if (*this > 1) {
    num -= 2 * denom;
  } else if (*this <= -1) {
    num += 2 * denom;
  }
  int32_t g = gcd(num, denom);
  num /= g;
  denom /= g;
  if (denom < 0) {
    num = -num;
    denom = -denom;
  }
}

double Rational::to_double() const {
  return zx::PI * (static_cast<float>(num)) / denom;
}
  
Rational &Rational::operator+=(const Rational &rhs) {
  num = num * rhs.denom + rhs.num * denom;
  denom *= rhs.denom;
  normalize();
  return *this;
}
Rational &Rational::operator+=(const int32_t rhs) {
  num = num + rhs * denom;
  normalize();
  return *this;
}

Rational &Rational::operator-=(const Rational &rhs) {
  num = num * rhs.denom - rhs.num * denom;
  denom *= rhs.denom;
  normalize();
  return *this;
}
Rational &Rational::operator-=(const int32_t rhs) {
  num = num + rhs * denom;
  normalize();
  return *this;
}

Rational &Rational::operator*=(const Rational &rhs) {
  num *= rhs.num;
  denom *= rhs.denom;
  this->normalize();
  return *this;
}
Rational &Rational::operator*=(const int32_t rhs) {
  num *= rhs;
  this->normalize();
  return *this;
}

Rational &Rational::operator/=(const Rational &rhs) {
  num *= rhs.denom;
  denom *= rhs.num;
  this->normalize();
  return *this;
}
Rational &Rational::operator/=(const int32_t rhs) {
  denom *= rhs;
  this->normalize();
  return *this;
}
} // namespace zx
