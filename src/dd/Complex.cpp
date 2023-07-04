#include "dd/Complex.hpp"

#include "dd/ComplexValue.hpp"

namespace dd {
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
Complex Complex::zero{&ComplexTable::zero, &ComplexTable::zero};
Complex Complex::one{&ComplexTable::one, &ComplexTable::zero};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

void Complex::setVal(const Complex& c) const {
  assert(!CTEntry::isNegativePointer(r));
  assert(!CTEntry::isNegativePointer(i));
  r->value = CTEntry::val(c.r);
  i->value = CTEntry::val(c.i);
}

bool Complex::exactlyZero() const {
  return CTEntry::exactlyZero(r) && CTEntry::exactlyZero(i);
}

bool Complex::exactlyOne() const {
  return CTEntry::exactlyOne(r) && CTEntry::exactlyZero(i);
}

bool Complex::approximatelyEquals(const Complex& c) const {
  return CTEntry::approximatelyEquals(r, c.r) &&
         CTEntry::approximatelyEquals(i, c.i);
}

bool Complex::approximatelyZero() const {
  return CTEntry::approximatelyZero(r) && CTEntry::approximatelyZero(i);
}

bool Complex::approximatelyOne() const {
  return CTEntry::approximatelyOne(r) && CTEntry::approximatelyZero(i);
}

bool Complex::operator==(const Complex& other) const {
  return r == other.r && i == other.i;
}

bool Complex::operator!=(const Complex& other) const {
  return !operator==(other);
}

std::string Complex::toString(bool formatted, int precision) const {
  return ComplexValue::toString(CTEntry::val(r), CTEntry::val(i), formatted,
                                precision);
}

void Complex::writeBinary(std::ostream& os) const {
  CTEntry::writeBinary(r, os);
  CTEntry::writeBinary(i, os);
}

std::ostream& operator<<(std::ostream& os, const Complex& c) {
  return os << c.toString();
}
} // namespace dd
