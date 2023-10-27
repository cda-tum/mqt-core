#include "dd/ComplexNumbers.hpp"

#include "dd/ComplexValue.hpp"

#include <cassert>
#include <cmath>

namespace dd {

void ComplexNumbers::setTolerance(fp tol) noexcept { RealNumber::eps = tol; }

void ComplexNumbers::add(Complex& r, const Complex& a,
                         const Complex& b) noexcept {
  assert(!r.exactlyZero());
  assert(!r.exactlyOne());
  assert(r.r != a.i && "r.r and a.i point to the same entry!");
  assert(r.i != a.r && "r.i and a.r point to the same entry!");
  assert(r.r != b.i && "r.r and b.i point to the same entry!");
  assert(r.i != b.r && "r.i and b.r point to the same entry!");
  r.r->value = RealNumber::val(a.r) + RealNumber::val(b.r);
  r.i->value = RealNumber::val(a.i) + RealNumber::val(b.i);
}

void ComplexNumbers::sub(Complex& r, const Complex& a,
                         const Complex& b) noexcept {
  assert(!r.exactlyZero());
  assert(!r.exactlyOne());
  assert(r.r != a.i && "r.r and a.i point to the same entry!");
  assert(r.i != a.r && "r.i and a.r point to the same entry!");
  assert(r.r != b.i && "r.r and b.i point to the same entry!");
  assert(r.i != b.r && "r.i and b.r point to the same entry!");
  r.r->value = RealNumber::val(a.r) - RealNumber::val(b.r);
  r.i->value = RealNumber::val(a.i) - RealNumber::val(b.i);
}

void ComplexNumbers::mul(Complex& r, const Complex& a,
                         const Complex& b) noexcept {
  assert(!r.exactlyZero());
  assert(!r.exactlyOne());
  assert(r.r != a.i && "r.r and a.i point to the same entry!");
  assert(r.i != a.r && "r.i and a.r point to the same entry!");
  assert(r.r != b.i && "r.r and b.i point to the same entry!");
  assert(r.i != b.r && "r.i and b.r point to the same entry!");
  const auto aVal = static_cast<ComplexValue>(a);
  const auto bVal = static_cast<ComplexValue>(b);
  const auto rVal = aVal * bVal;
  r.r->value = rVal.r;
  r.i->value = rVal.i;
}

void ComplexNumbers::div(Complex& r, const Complex& a,
                         const Complex& b) noexcept {
  assert(!r.exactlyZero());
  assert(!r.exactlyOne());
  assert(r.r != a.i && "r.r and a.i point to the same entry!");
  assert(r.i != a.r && "r.i and a.r point to the same entry!");
  assert(r.r != b.i && "r.r and b.i point to the same entry!");
  assert(r.i != b.r && "r.i and b.r point to the same entry!");
  const auto aVal = static_cast<ComplexValue>(a);
  const auto bVal = static_cast<ComplexValue>(b);
  const auto rVal = aVal / bVal;
  r.r->value = rVal.r;
  r.i->value = rVal.i;
}

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
  return cacheManager->getStats().numUsed;
}

std::size_t ComplexNumbers::realCount() const noexcept {
  return uniqueTable->getStats().numEntries;
}

} // namespace dd
