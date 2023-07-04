#include "dd/ComplexNumbers.hpp"

#include <cassert>
#include <cmath>

namespace dd {
void ComplexNumbers::clear() {
  complexTable.clear();
  complexCache.clear();
}

void ComplexNumbers::setTolerance(fp tol) { ComplexTable::setTolerance(tol); }

void ComplexNumbers::add(Complex& r, const Complex& a, const Complex& b) {
  assert(r != Complex::zero);
  assert(r != Complex::one);
  r.r->value = CTEntry::val(a.r) + CTEntry::val(b.r);
  r.i->value = CTEntry::val(a.i) + CTEntry::val(b.i);
}

void ComplexNumbers::sub(Complex& r, const Complex& a, const Complex& b) {
  assert(r != Complex::zero);
  assert(r != Complex::one);
  r.r->value = CTEntry::val(a.r) - CTEntry::val(b.r);
  r.i->value = CTEntry::val(a.i) - CTEntry::val(b.i);
}

void ComplexNumbers::mul(Complex& r, const Complex& a, const Complex& b) {
  assert(r != Complex::zero);
  assert(r != Complex::one);
  if (a.approximatelyOne()) {
    r.setVal(b);
  } else if (b.approximatelyOne()) {
    r.setVal(a);
  } else if (a.approximatelyZero() || b.approximatelyZero()) {
    r.r->value = 0.;
    r.i->value = 0.;
  } else {
    const auto ar = CTEntry::val(a.r);
    const auto ai = CTEntry::val(a.i);
    const auto br = CTEntry::val(b.r);
    const auto bi = CTEntry::val(b.i);

    r.r->value = ar * br - ai * bi;
    r.i->value = ar * bi + ai * br;
  }
}

void ComplexNumbers::div(Complex& r, const Complex& a, const Complex& b) {
  assert(r != Complex::zero);
  assert(r != Complex::one);
  if (a.approximatelyEquals(b)) {
    r.r->value = 1.;
    r.i->value = 0.;
  } else if (b.approximatelyOne()) {
    r.setVal(a);
  } else {
    const auto ar = CTEntry::val(a.r);
    const auto ai = CTEntry::val(a.i);
    const auto br = CTEntry::val(b.r);
    const auto bi = CTEntry::val(b.i);

    const auto cmag = br * br + bi * bi;

    r.r->value = (ar * br + ai * bi) / cmag;
    r.i->value = (ai * br - ar * bi) / cmag;
  }
}

fp ComplexNumbers::mag2(const Complex& a) {
  auto ar = CTEntry::val(a.r);
  auto ai = CTEntry::val(a.i);

  return ar * ar + ai * ai;
}

fp ComplexNumbers::mag(const Complex& a) { return std::sqrt(mag2(a)); }

fp ComplexNumbers::arg(const Complex& a) {
  auto ar = CTEntry::val(a.r);
  auto ai = CTEntry::val(a.i);
  return std::atan2(ai, ar);
}

Complex ComplexNumbers::conj(const Complex& a) {
  return {a.r, CTEntry::flipPointerSign(a.i)};
}

Complex ComplexNumbers::neg(const Complex& a) {
  return {CTEntry::flipPointerSign(a.r), CTEntry::flipPointerSign(a.i)};
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

Complex ComplexNumbers::lookup(const Complex& c) {
  if (c == Complex::zero) {
    return Complex::zero;
  }
  if (c == Complex::one) {
    return Complex::one;
  }

  const auto valr = CTEntry::val(c.r);
  const auto vali = CTEntry::val(c.i);
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
    const auto absr = std::abs(r);
    // if absolute value is close enough to zero, just return the zero entry
    // (avoiding -0.0)
    if (absr < ComplexTable::tolerance()) {
      ret.r = &ComplexTable::zero;
    } else {
      ret.r = CTEntry::getNegativePointer(complexTable.lookup(absr));
    }
  } else {
    ret.r = complexTable.lookup(r);
  }

  if (const auto signI = std::signbit(i); signI) {
    const auto absi = std::abs(i);
    // if absolute value is close enough to zero, just return the zero entry
    // (avoiding -0.0)
    if (absi < ComplexTable::tolerance()) {
      ret.i = &ComplexTable::zero;
    } else {
      ret.i = CTEntry::getNegativePointer(complexTable.lookup(absi));
    }
  } else {
    ret.i = complexTable.lookup(i);
  }

  return ret;
}

void ComplexNumbers::incRef(const Complex& c) {
  if (!isStaticComplex(c)) {
    ComplexTable::incRef(c.r);
    ComplexTable::incRef(c.i);
  }
}

void ComplexNumbers::decRef(const Complex& c) {
  if (!isStaticComplex(c)) {
    ComplexTable::decRef(c.r);
    ComplexTable::decRef(c.i);
  }
}

std::size_t ComplexNumbers::garbageCollect(bool force) {
  return complexTable.garbageCollect(force);
}

Complex ComplexNumbers::getTemporary() {
  return complexCache.getTemporaryComplex();
}

Complex ComplexNumbers::getTemporary(const fp& r, const fp& i) {
  auto c = complexCache.getTemporaryComplex();
  c.r->value = r;
  c.i->value = i;
  return c;
}

Complex ComplexNumbers::getTemporary(const ComplexValue& c) {
  return getTemporary(c.r, c.i);
}

Complex ComplexNumbers::getCached() { return complexCache.getCachedComplex(); }

Complex ComplexNumbers::getCached(const fp& r, const fp& i) {
  auto c = complexCache.getCachedComplex();
  c.r->value = r;
  c.i->value = i;
  return c;
}

Complex ComplexNumbers::getCached(const ComplexValue& c) {
  return getCached(c.r, c.i);
}

Complex ComplexNumbers::getCached(const std::complex<fp>& c) {
  return getCached(c.real(), c.imag());
}

void ComplexNumbers::returnToCache(Complex& c) {
  complexCache.returnToCache(c);
}

std::size_t ComplexNumbers::cacheCount() const {
  return complexCache.getCount();
}

} // namespace dd
