#include "dd/Complex.hpp"

#include "dd/ComplexValue.hpp"
#include "dd/RealNumber.hpp"

#include <cassert>

namespace dd {

void Complex::setVal(const Complex& c) const noexcept {
  assert(!RealNumber::isNegativePointer(r));
  assert(!RealNumber::isNegativePointer(i));
  r->value = RealNumber::val(c.r);
  i->value = RealNumber::val(c.i);
}

bool Complex::approximatelyEquals(const Complex& c) const noexcept {
  return RealNumber::approximatelyEquals(r, c.r) &&
         RealNumber::approximatelyEquals(i, c.i);
}

bool Complex::approximatelyZero() const noexcept {
  return RealNumber::approximatelyZero(r) && RealNumber::approximatelyZero(i);
}

bool Complex::approximatelyOne() const noexcept {
  return RealNumber::approximatelyOne(r) && RealNumber::approximatelyZero(i);
}

bool Complex::operator==(const Complex& other) const noexcept {
  return r == other.r && i == other.i;
}

bool Complex::operator!=(const Complex& other) const noexcept {
  return !operator==(other);
}

std::string Complex::toString(bool formatted, int precision) const {
  return ComplexValue::toString(RealNumber::val(r), RealNumber::val(i),
                                formatted, precision);
}

void Complex::writeBinary(std::ostream& os) const {
  RealNumber::writeBinary(r, os);
  RealNumber::writeBinary(i, os);
}

Complex::operator std::complex<fp>() const noexcept {
  return {RealNumber::val(r), RealNumber::val(i)};
}

Complex::operator ComplexValue() const noexcept {
  return ComplexValue{RealNumber::val(r), RealNumber::val(i)};
}

std::ostream& operator<<(std::ostream& os, const Complex& c) {
  return os << c.toString();
}

ComplexValue operator*(const Complex& c1, const ComplexValue& c2) {
  return static_cast<ComplexValue>(c1) * c2;
}
ComplexValue operator*(const ComplexValue& c1, const Complex& c2) {
  return c1 * static_cast<ComplexValue>(c2);
}
ComplexValue operator*(const Complex& c1, const Complex& c2) {
  return static_cast<ComplexValue>(c1) * static_cast<ComplexValue>(c2);
}
ComplexValue operator*(const Complex& c1, const fp real) {
  return static_cast<ComplexValue>(c1) * real;
}
ComplexValue operator*(const fp real, const Complex& c1) { return c1 * real; }

ComplexValue operator/(const Complex& c1, const ComplexValue& c2) {
  return static_cast<ComplexValue>(c1) / c2;
}
ComplexValue operator/(const ComplexValue& c1, const Complex& c2) {
  return c1 / static_cast<ComplexValue>(c2);
}
ComplexValue operator/(const Complex& c1, const Complex& c2) {
  return static_cast<ComplexValue>(c1) / static_cast<ComplexValue>(c2);
}
ComplexValue operator/(const Complex& c1, const fp real) {
  return static_cast<ComplexValue>(c1) / real;
}
} // namespace dd
