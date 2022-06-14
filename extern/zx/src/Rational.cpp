#include "Rational.hpp"
#include "Definitions.hpp"
#include <cmath>
#include <gmpxx.h>

namespace zx {
mpz_class gcd(mpz_class a, mpz_class b) {
  while (b != 0) {
    mpz_class r = 0;
    mpz_mod(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
    a = b;
    b = r;
  }
  return a;
}

PiRational::PiRational(double val) : num(0), denom(1) {
  if (std::abs(val) < PARAMETER_TOLERANCE)
    return;
  
  double mult_pi = PI / val;
  double nearest = std::round(mult_pi);
  if (std::abs(nearest - mult_pi) < PARAMETER_TOLERANCE) {
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

  mpz_class gcd_ = gcd(std::round(frac * MAX_DENOM), MAX_DENOM);

  denom = MAX_DENOM / gcd_;
  num = round(frac * MAX_DENOM) / gcd_;
  if(denom < 0) {
    num = -num;
    denom = -denom;
  }
}

void PiRational::normalize() {
  if (*this > 1) {
    num -= 2 * denom;
  } else if (*this <= -1) {
    num += 2 * denom;
  }
  if(num == 0) {
    denom = 1;
    return;
  }
    
  mpz_class g = gcd(num, denom);
  num /= g;
  denom /= g;

  if (denom < 0) {
    num = -num;
    denom = -denom;
  }
}

<<<<<<< HEAD

=======
>>>>>>> ea18283 (Removed QFR as dependency)
// double PiRational::to_double() const {
//   return zx::PI * (static_cast<float>(num)) / denom;
// }
  
PiRational &PiRational::operator+=(const PiRational &rhs) {
  num = num * rhs.denom + rhs.num * denom;
  denom *= rhs.denom;
  normalize();
  return *this;
}
<<<<<<< HEAD

=======
>>>>>>> ea18283 (Removed QFR as dependency)
PiRational &PiRational::operator+=(const int64_t rhs) {
  num = num + rhs * denom;
  normalize();
  return *this;
}

PiRational &PiRational::operator-=(const PiRational &rhs) {
  num = num * rhs.denom - rhs.num * denom;
  denom *= rhs.denom;
  normalize();
  return *this;
}

PiRational &PiRational::operator-=(const int64_t rhs) {
  num = num + rhs * denom;
  normalize();
  return *this;
}

PiRational &PiRational::operator*=(const PiRational &rhs) {
  num *= rhs.num;
  denom *= rhs.denom;
  this->normalize();
  return *this;
}

PiRational &PiRational::operator*=(const int64_t rhs) {
  num *= rhs;
  this->normalize();
  return *this;
}

PiRational &PiRational::operator/=(const PiRational &rhs) {
  num *= rhs.denom;
  denom *= rhs.num;
  this->normalize();
  return *this;
}

PiRational &PiRational::operator/=(const int64_t rhs) {
  denom *= rhs;
  this->normalize();
  return *this;
}
} // namespace zx
