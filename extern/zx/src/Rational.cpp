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
  if (std::abs(val) < qc::PARAMETER_TOLERANCE)
    return;
  
  double mult_pi = PI / val;
  double nearest = std::round(mult_pi);
  if (std::abs(nearest - mult_pi) < qc::PARAMETER_TOLERANCE) {
    denom = static_cast<int>(nearest);
    num = 1;
    if(denom < 0) {
      num = -1;
      denom = - denom;
    }

    return;
  }

  val /= PI;
  val -= 2*static_cast<int>(val/2);
  if (val > 1) {
    val -= 2;
  } else if (val <= -1) {
    val += 2;
  }

  // double integral = val >= 0.0 ? std::floor(val) : std::ceil(val);
  // double frac = val - integral;
  double frac = val;

  long gcd_ = gcd(std::round(frac * MAX_DENOM), MAX_DENOM);

  denom = MAX_DENOM / gcd_;
  num = round(frac * MAX_DENOM) / gcd_;
  if(denom < 0) {
    num = -num;
    denom = -denom;
  }
}

void Rational::normalize() {
  if (*this > 1) {
    num -= 2 * denom;
  } else if (*this <= -1) {
    num += 2 * denom;
  }
  if(num == 0) {
    denom = 1;
    return;
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
