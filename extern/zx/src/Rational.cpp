#include "Definitions.hpp"
#include "Rational.hpp"
#include <cmath>

long gcd(long a, long b) {
  long r;

  while (b != 0) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}

Rational::Rational(float val) : num(0), denom(1) {
  val /= PI;
  
  double integral = std::floor(val);
  double frac = val - integral;


  long gcd_ = gcd(std::round(frac * MAX_DENOM), MAX_DENOM);

  denom = MAX_DENOM / gcd_;
  num = round(frac * MAX_DENOM) / gcd_;
}

void Rational::normalize() {
  if (*this > 1) {
    *this -= *this - 1;
  } else if (*this <= -1) {
    *this += -1 - *this;
  }
  int32_t g = gcd(num, denom);
  num /= g;
  denom /= g;
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
