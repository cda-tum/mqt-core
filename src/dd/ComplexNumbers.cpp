#include "dd/ComplexNumbers.hpp"

#include <cassert>
#include <cmath>

namespace dd {

void ComplexNumbers::setTolerance(fp tol) noexcept { RealNumber::eps = tol; }

void ComplexNumbers::add(Complex& r, const Complex& a,
                         const Complex& b) noexcept {
  assert(r != Complex::zero);
  assert(r != Complex::one);
  assert(r.r != a.i && "r.r and a.i point to the same entry!");
  assert(r.i != a.r && "r.i and a.r point to the same entry!");
  assert(r.r != b.i && "r.r and b.i point to the same entry!");
  assert(r.i != b.r && "r.i and b.r point to the same entry!");
  r.r->value = RealNumber::val(a.r) + RealNumber::val(b.r);
  r.i->value = RealNumber::val(a.i) + RealNumber::val(b.i);
}

void ComplexNumbers::sub(Complex& r, const Complex& a,
                         const Complex& b) noexcept {
  assert(r != Complex::zero);
  assert(r != Complex::one);
  assert(r.r != a.i && "r.r and a.i point to the same entry!");
  assert(r.i != a.r && "r.i and a.r point to the same entry!");
  assert(r.r != b.i && "r.r and b.i point to the same entry!");
  assert(r.i != b.r && "r.i and b.r point to the same entry!");
  r.r->value = RealNumber::val(a.r) - RealNumber::val(b.r);
  r.i->value = RealNumber::val(a.i) - RealNumber::val(b.i);
}

void ComplexNumbers::mul(Complex& r, const Complex& a,
                         const Complex& b) noexcept {
  assert(r != Complex::zero);
  assert(r != Complex::one);
  assert(r.r != a.i && "r.r and a.i point to the same entry!");
  assert(r.i != a.r && "r.i and a.r point to the same entry!");
  assert(r.r != b.i && "r.r and b.i point to the same entry!");
  assert(r.i != b.r && "r.i and b.r point to the same entry!");
  if (a.approximatelyOne()) {
    r.setVal(b);
  } else if (b.approximatelyOne()) {
    r.setVal(a);
  } else if (a.approximatelyZero() || b.approximatelyZero()) {
    r.r->value = 0.;
    r.i->value = 0.;
  } else {
    const auto ar = RealNumber::val(a.r);
    const auto ai = RealNumber::val(a.i);
    const auto br = RealNumber::val(b.r);
    const auto bi = RealNumber::val(b.i);

    r.r->value = ar * br - ai * bi;
    r.i->value = ar * bi + ai * br;
  }
}

void ComplexNumbers::div(Complex& r, const Complex& a,
                         const Complex& b) noexcept {
  assert(r != Complex::zero);
  assert(r != Complex::one);
  assert(r.r != a.i && "r.r and a.i point to the same entry!");
  assert(r.i != a.r && "r.i and a.r point to the same entry!");
  assert(r.r != b.i && "r.r and b.i point to the same entry!");
  assert(r.i != b.r && "r.i and b.r point to the same entry!");
  if (a.approximatelyEquals(b)) {
    r.r->value = 1.;
    r.i->value = 0.;
  } else if (b.approximatelyOne()) {
    r.setVal(a);
  } else {
    const auto ar = RealNumber::val(a.r);
    const auto ai = RealNumber::val(a.i);
    const auto br = RealNumber::val(b.r);
    const auto bi = RealNumber::val(b.i);

    const auto cmag = br * br + bi * bi;

    r.r->value = (ar * br + ai * bi) / cmag;
    r.i->value = (ai * br - ar * bi) / cmag;
  }
}

fp ComplexNumbers::mag2(const Complex& a) noexcept {
  auto ar = RealNumber::val(a.r);
  auto ai = RealNumber::val(a.i);

  return ar * ar + ai * ai;
}

fp ComplexNumbers::mag(const Complex& a) noexcept { return std::sqrt(mag2(a)); }

fp ComplexNumbers::arg(const Complex& a) noexcept {
  auto ar = RealNumber::val(a.r);
  auto ai = RealNumber::val(a.i);
  return std::atan2(ai, ar);
}

Complex ComplexNumbers::conj(const Complex& a) noexcept {
  return {a.r, RealNumber::flipPointerSign(a.i)};
}

Complex ComplexNumbers::neg(const Complex& a) noexcept {
  return {RealNumber::flipPointerSign(a.r), RealNumber::flipPointerSign(a.i)};
}

Complex ComplexNumbers::addCached(const Complex& a, const Complex& b) {
  auto c = getCached();
  add(c, a, b);
  return c;
}

Complex ComplexNumbers::subCached(const Complex& a, const Complex& b) {
  auto c = getCached();
  sub(c, a, b);
  return c;
}

Complex ComplexNumbers::mulCached(const Complex& a, const Complex& b) {
  auto c = getCached();
  mul(c, a, b);
  return c;
}

Complex ComplexNumbers::divCached(const Complex& a, const Complex& b) {
  auto c = getCached();
  div(c, a, b);
  return c;
}

Complex ComplexNumbers::addTemp(const dd::Complex& a, const dd::Complex& b) {
  auto c = getTemporary();
  add(c, a, b);
  return c;
}

Complex ComplexNumbers::subTemp(const dd::Complex& a, const dd::Complex& b) {
  auto c = getTemporary();
  sub(c, a, b);
  return c;
}

Complex ComplexNumbers::mulTemp(const dd::Complex& a, const dd::Complex& b) {
  auto c = getTemporary();
  mul(c, a, b);
  return c;
}

Complex ComplexNumbers::divTemp(const dd::Complex& a, const dd::Complex& b) {
  auto c = getTemporary();
  div(c, a, b);
  return c;
}

Complex ComplexNumbers::lookup(const Complex& c, const bool cached) {
  if (isStaticComplex(c)) {
    return c;
  }

  const auto valr = RealNumber::val(c.r);
  const auto vali = RealNumber::val(c.i);

  if (cached) {
    returnToCache(c);
  }

  return lookup(valr, vali);
}

Complex ComplexNumbers::lookup(const std::complex<fp>& c) {
  return lookup(c.real(), c.imag());
}

Complex ComplexNumbers::lookup(const ComplexValue& c) {
  return lookup(c.r, c.i);
}

Complex ComplexNumbers::lookup(const fp r, const fp i) {
  Complex ret{};

  if (const auto signR = std::signbit(r); signR) {
    // if absolute value is close enough to zero, just return the zero entry
    // (avoiding -0.0)
    if (RealNumber::approximatelyZero(r)) {
      ret.r = &constants::zero;
    } else {
      ret.r = RealNumber::getNegativePointer(uniqueTable->lookup(std::abs(r)));
    }
  } else {
    ret.r = uniqueTable->lookup(r);
  }

  if (const auto signI = std::signbit(i); signI) {
    // if absolute value is close enough to zero, just return the zero entry
    // (avoiding -0.0)
    if (RealNumber::approximatelyZero(i)) {
      ret.i = &constants::zero;
    } else {
      ret.i = RealNumber::getNegativePointer(uniqueTable->lookup(std::abs(i)));
    }
  } else {
    ret.i = uniqueTable->lookup(i);
  }

  return ret;
}

void ComplexNumbers::incRef(const Complex& c) noexcept {
  if (!isStaticComplex(c)) {
    RealNumber::incRef(c.r);
    RealNumber::incRef(c.i);
  }
}

void ComplexNumbers::decRef(const Complex& c) noexcept {
  if (!isStaticComplex(c)) {
    RealNumber::decRef(c.r);
    RealNumber::decRef(c.i);
  }
}

Complex ComplexNumbers::getTemporary() {
  const auto [rv, iv] = cacheManager->getTemporaryPair();
  return {rv, iv};
}

Complex ComplexNumbers::getTemporary(const fp r, const fp i) {
  const auto [rv, iv] = cacheManager->getTemporaryPair();
  rv->value = r;
  iv->value = i;
  return {rv, iv};
}

Complex ComplexNumbers::getTemporary(const ComplexValue& c) {
  return getTemporary(c.r, c.i);
}

Complex ComplexNumbers::getTemporary(const Complex& c) {
  return getTemporary(RealNumber::val(c.r), RealNumber::val(c.i));
}

Complex ComplexNumbers::getCached() {
  const auto [rv, iv] = cacheManager->getPair();
  return {rv, iv};
}

Complex ComplexNumbers::getCached(const fp r, const fp i) {
  const auto [rv, iv] = getCached();
  rv->value = r;
  iv->value = i;
  return {rv, iv};
}

Complex ComplexNumbers::getCached(const Complex& c) {
  return getCached(RealNumber::val(c.r), RealNumber::val(c.i));
}

Complex ComplexNumbers::getCached(const ComplexValue& c) {
  return getCached(c.r, c.i);
}

Complex ComplexNumbers::getCached(const std::complex<fp>& c) {
  return getCached(c.real(), c.imag());
}

void ComplexNumbers::returnToCache(const Complex& c) noexcept {
  if (!constants::isStaticNumber(c.i)) {
    cacheManager->returnEntry(c.i);
  }
  if (!constants::isStaticNumber(c.r)) {
    cacheManager->returnEntry(c.r);
  }
}

std::size_t ComplexNumbers::cacheCount() const noexcept {
  return cacheManager->getUsedCount();
}

std::size_t ComplexNumbers::realCount() const noexcept {
  return uniqueTable->getStats().entryCount;
}

} // namespace dd
