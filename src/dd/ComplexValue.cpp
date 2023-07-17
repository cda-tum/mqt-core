#include "dd/ComplexValue.hpp"

#include "dd/RealNumber.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace dd {
bool ComplexValue::operator==(const ComplexValue& other) const noexcept {
  // NOLINTNEXTLINE(clang-diagnostic-float-equal)
  return r == other.r && i == other.i;
}

bool ComplexValue::operator!=(const ComplexValue& other) const noexcept {
  return !operator==(other);
}

bool ComplexValue::approximatelyEquals(const ComplexValue& c) const noexcept {
  return RealNumber::approximatelyEquals(r, c.r) &&
         RealNumber::approximatelyEquals(i, c.i);
}

bool ComplexValue::approximatelyZero() const noexcept {
  return RealNumber::approximatelyZero(r) && RealNumber::approximatelyZero(i);
}

bool ComplexValue::approximatelyOne() const noexcept {
  return RealNumber::approximatelyOne(r) && RealNumber::approximatelyZero(i);
}

void ComplexValue::writeBinary(std::ostream& os) const {
  os.write(reinterpret_cast<const char*>(&r), sizeof(decltype(r)));
  os.write(reinterpret_cast<const char*>(&i), sizeof(decltype(i)));
}

void ComplexValue::readBinary(std::istream& is) {
  is.read(reinterpret_cast<char*>(&r), sizeof(decltype(r)));
  is.read(reinterpret_cast<char*>(&i), sizeof(decltype(i)));
}

void ComplexValue::fromString(const std::string& realStr, std::string imagStr) {
  r = realStr.empty() ? 0. : std::stod(realStr);

  imagStr.erase(remove(imagStr.begin(), imagStr.end(), ' '), imagStr.end());
  imagStr.erase(remove(imagStr.begin(), imagStr.end(), 'i'), imagStr.end());
  if (imagStr == "+" || imagStr == "-") {
    imagStr = imagStr + "1";
  }
  i = imagStr.empty() ? 0. : std::stod(imagStr);
}

std::pair<std::uint64_t, std::uint64_t>
ComplexValue::getLowestFraction(const double x,
                                const std::uint64_t maxDenominator) {
  assert(x >= 0.);

  std::pair<std::uint64_t, std::uint64_t> lowerBound{0U, 1U};
  std::pair<std::uint64_t, std::uint64_t> upperBound{1U, 0U};

  while ((lowerBound.second <= maxDenominator) &&
         (upperBound.second <= maxDenominator)) {
    auto num = lowerBound.first + upperBound.first;
    auto den = lowerBound.second + upperBound.second;
    auto median = static_cast<fp>(num) / static_cast<fp>(den);
    if (std::abs(x - median) <= RealNumber::eps) {
      if (den <= maxDenominator) {
        return std::pair{num, den};
      }
      if (upperBound.second > lowerBound.second) {
        return upperBound;
      }
      return lowerBound;
    }
    if (x > median) {
      lowerBound = {num, den};
    } else {
      upperBound = {num, den};
    }
  }
  if (lowerBound.second > maxDenominator) {
    return upperBound;
  }
  return lowerBound;
}

void ComplexValue::printFormatted(std::ostream& os, fp num, bool imaginary) {
  if (RealNumber::approximatelyZero(num)) {
    os << (std::signbit(num) ? "-" : "+") << "0" << (imaginary ? "i" : "");
    return;
  }

  const auto absnum = std::abs(num);
  auto fraction = getLowestFraction(absnum);
  auto approx =
      static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);

  // suitable fraction a/b found
  if (const auto error = absnum - approx;
      RealNumber::approximatelyZero(error)) {
    const std::string sign = std::signbit(num) ? "-" : (imaginary ? "+" : "");

    if (fraction.first == 1U && fraction.second == 1U) {
      os << sign << (imaginary ? "i" : "1");
    } else if (fraction.second == 1U) {
      os << sign << fraction.first << (imaginary ? "i" : "");
    } else if (fraction.first == 1U) {
      os << sign << (imaginary ? "i" : "1") << "/" << fraction.second;
    } else {
      os << sign << fraction.first << (imaginary ? "i" : "") << "/"
         << fraction.second;
    }

    return;
  }

  const auto abssqrt = absnum / SQRT2_2;
  fraction = getLowestFraction(abssqrt);
  approx = static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
  // suitable fraction a/(b * sqrt(2)) found
  if (const auto error = abssqrt - approx;
      RealNumber::approximatelyZero(error)) {
    const std::string sign = std::signbit(num) ? "-" : (imaginary ? "+" : "");

    if (fraction.first == 1U && fraction.second == 1U) {
      os << sign << (imaginary ? "i" : "1") << "/√2";
    } else if (fraction.second == 1U) {
      os << sign << fraction.first << (imaginary ? "i" : "") << "/√2";
    } else if (fraction.first == 1U) {
      os << sign << (imaginary ? "i" : "1") << "/(" << fraction.second << "√2)";
    } else {
      os << sign << fraction.first << (imaginary ? "i" : "") << "/("
         << fraction.second << "√2)";
    }
    return;
  }

  const auto abspi = absnum / PI;
  fraction = getLowestFraction(abspi);
  approx = static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
  // suitable fraction a/b π found
  if (const auto error = abspi - approx; RealNumber::approximatelyZero(error)) {
    const std::string sign = std::signbit(num) ? "-" : (imaginary ? "+" : "");
    const std::string imagUnit = imaginary ? "i" : "";

    if (fraction.first == 1U && fraction.second == 1U) {
      os << sign << "π" << imagUnit;
    } else if (fraction.second == 1U) {
      os << sign << fraction.first << "π" << imagUnit;
    } else if (fraction.first == 1U) {
      os << sign << "π" << imagUnit << "/" << fraction.second;
    } else {
      os << sign << fraction.first << "π" << imagUnit << "/" << fraction.second;
    }
    return;
  }

  if (imaginary) { // default
    os << (std::signbit(num) ? "" : "+") << num << "i";
  } else {
    os << num;
  }
}

std::string ComplexValue::toString(const fp& real, const fp& imag,
                                   bool formatted, int precision) {
  std::ostringstream ss{};

  if (precision >= 0) {
    ss << std::setprecision(precision);
  }
  if (RealNumber::approximatelyZero(real) &&
      RealNumber::approximatelyZero(imag)) {
    return "0";
  }

  if (!RealNumber::approximatelyZero(real)) {
    if (formatted) {
      printFormatted(ss, real);
    } else {
      ss << real;
    }
  }
  if (!RealNumber::approximatelyZero(imag)) {
    if (formatted) {
      if (RealNumber::approximatelyEquals(real, imag)) {
        ss << "(1+i)";
        return ss.str();
      }
      if (RealNumber::approximatelyEquals(real, -imag)) {
        ss << "(1-i)";
        return ss.str();
      }
      printFormatted(ss, imag, true);
    } else {
      if (RealNumber::approximatelyZero(real)) {
        ss << imag;
      } else {
        if (imag > 0.) {
          ss << "+";
        }
        ss << imag;
      }
      ss << "i";
    }
  }

  return ss.str();
}

ComplexValue& ComplexValue::operator+=(const ComplexValue& rhs) noexcept {
  r += rhs.r;
  i += rhs.i;
  return *this;
}

ComplexValue operator+(ComplexValue lhs, const ComplexValue& rhs) noexcept {
  lhs += rhs;
  return lhs;
}

std::ostream& operator<<(std::ostream& os, const ComplexValue& c) {
  return os << ComplexValue::toString(c.r, c.i);
}
} // namespace dd

namespace std {
std::size_t
hash<dd::ComplexValue>::operator()(const dd::ComplexValue& c) const noexcept {
  auto h1 = dd::murmur64(
      static_cast<std::size_t>(std::round(c.r / dd::RealNumber::eps)));
  auto h2 = dd::murmur64(
      static_cast<std::size_t>(std::round(c.i / dd::RealNumber::eps)));
  return dd::combineHash(h1, h2);
}
} // namespace std
