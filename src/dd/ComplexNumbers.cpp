#include "dd/ComplexNumbers.hpp"

#include "dd/ComplexValue.hpp"
#include "dd/RealNumber.hpp"

#include <cmath>
#include <complex>

namespace dd {

void ComplexNumbers::setTolerance(fp tol) noexcept { RealNumber::eps = tol; }

fp ComplexNumbers::mag2(const Complex& a) noexcept {
  return static_cast<ComplexValue>(a).mag2();
}

fp ComplexNumbers::mag(const Complex& a) noexcept {
  return static_cast<ComplexValue>(a).mag();
}

fp ComplexNumbers::arg(const Complex& a) noexcept {
  const auto val = static_cast<ComplexValue>(a);
  return std::atan2(val.i, val.r);
}

Complex ComplexNumbers::conj(const Complex& a) noexcept {
  return {a.r, RealNumber::flipPointerSign(a.i)};
}

Complex ComplexNumbers::neg(const Complex& a) noexcept {
  return {RealNumber::flipPointerSign(a.r), RealNumber::flipPointerSign(a.i)};
}

Complex ComplexNumbers::lookup(const Complex& c) {
  if (isStaticComplex(c)) {
    return c;
  }

  const auto valr = RealNumber::val(c.r);
  const auto vali = RealNumber::val(c.i);
  return lookup(valr, vali);
}

void ComplexNumbers::incRef(const Complex& c) const noexcept {
  uniqueTable->incRef(c.r);
  uniqueTable->incRef(c.i);
}

void ComplexNumbers::decRef(const Complex& c) const noexcept {
  uniqueTable->decRef(c.r);
  uniqueTable->decRef(c.i);
}

Complex ComplexNumbers::lookup(const std::complex<fp>& c) {
  return lookup(c.real(), c.imag());
}

Complex ComplexNumbers::lookup(const ComplexValue& c) {
  return lookup(c.r, c.i);
}

Complex ComplexNumbers::lookup(const fp r) {
  return {uniqueTable->lookup(r), &constants::zero};
}

Complex ComplexNumbers::lookup(const fp r, const fp i) {
  return {uniqueTable->lookup(r), uniqueTable->lookup(i)};
}

std::size_t ComplexNumbers::realCount() const noexcept {
  return uniqueTable->getStats().numEntries;
}

} // namespace dd
