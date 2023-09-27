#pragma once

#include "dd/DDDefinitions.hpp"

#include <complex>
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>

namespace dd {
/// A complex number represented by two floating point values.
struct ComplexValue {
  /// real part
  fp r;
  /// imaginary part
  fp i;

  /**
   * @brief Check for exact equality.
   * @param other The other complex number to compare to.
   * @returns True if the complex numbers are exactly equal, false otherwise.
   */
  [[nodiscard]] bool operator==(const ComplexValue& other) const noexcept;

  /// @see operator==
  [[nodiscard]] bool operator!=(const ComplexValue& other) const noexcept;

  /**
   * @brief Check whether the complex number is approximately equal to the
   * given complex number.
   * @param c The complex number to compare to.
   * @returns True if the complex number is approximately equal to the given
   * complex number, false otherwise.
   * @see CTEntry::approximatelyEquals
   */
  [[nodiscard]] bool approximatelyEquals(const ComplexValue& c) const noexcept;

  /**
   * @brief Check whether the complex number is approximately equal to zero.
   * @returns True if the complex number is approximately equal to zero, false
   * otherwise.
   * @see CTEntry::approximatelyZero
   */
  [[nodiscard]] bool approximatelyZero() const noexcept;

  /**
   * @brief Check whether the complex number is approximately equal to one.
   * @returns True if the complex number is approximately equal to one, false
   * otherwise.
   * @see CTEntry::approximatelyOne
   * @see CTEntry::approximatelyZero
   */
  [[nodiscard]] bool approximatelyOne() const noexcept;

  /**
   * @brief Write a binary representation of the complex number to the given
   * output stream.
   * @param os The output stream to write to.
   */
  void writeBinary(std::ostream& os) const;

  /**
   * @brief Read a binary representation of the complex number from the given
   * input stream.
   * @param is The input stream to read from.
   */
  void readBinary(std::istream& is);

  /**
   * @brief Construct a complex number from a string.
   * @param realStr The string representation of the real part.
   * @param imagStr The string representation of the imaginary part.
   */
  void fromString(const std::string& realStr, std::string imagStr);

  /**
   * @brief Get the closest fraction to the given number.
   * @param x The number to approximate.
   * @param maxDenominator The maximum denominator to use.
   * @returns The closest fraction to the given number as a pair of numerator
   * and denominator.
   */
  static std::pair<std::uint64_t, std::uint64_t>
  getLowestFraction(double x, std::uint64_t maxDenominator = 1U << 10);

  /**
   * @brief Pretty print the given real number to the given output stream.
   * @param os The output stream to write to.
   * @param num The number to print.
   * @param imaginary Whether the number is imaginary.
   */
  static void printFormatted(std::ostream& os, fp num, bool imaginary = false);

  /**
   * @brief Print the given complex number to the given output stream.
   * @param real The real part of the complex number.
   * @param imag The imaginary part of the complex number.
   * @param formatted Whether to pretty print the number.
   * @param precision The precision to use for printing numbers..
   * @returns The string representation of the complex number.
   */
  static std::string toString(const fp& real, const fp& imag,
                              bool formatted = true, int precision = -1);

  /// Automatically convert to std::complex<dd::fp>
  explicit operator auto() const noexcept { return std::complex<dd::fp>{r, i}; }

  /// In-place addition of two complex numbers
  ComplexValue& operator+=(const ComplexValue& rhs) noexcept;

  /// Addition of two complex numbers
  friend ComplexValue operator+(ComplexValue lhs,
                                const ComplexValue& rhs) noexcept;
};

/**
 * @brief Print a complex value to the given output stream.
 * @param os The output stream to write to.
 * @param c The complex value to print.
 * @returns The output stream.
 */
std::ostream& operator<<(std::ostream& os, const ComplexValue& c);
} // namespace dd

namespace std {
/// Hash function for complex values
template <> struct hash<dd::ComplexValue> {
  /**
   * @brief Compute the hash value for the given complex value.
   * @details The hash value is computed by scaling the real and imaginary part
   * by the tolerance of the complex table, rounding the result to the nearest
   * integer and computing the hash value of the resulting pair of integers.
   * @param c The complex value to compute the hash value for.
   * @returns The hash value for the given complex value.
   * @note It is rather hard to define good hash functions for floating point
   * numbers. This hash function is not perfect, but it is fast and should
   * provide a good distribution of hash values. Furthermore, two floating point
   * numbers that are within the tolerance of the complex table will always
   * produce the same hash value.
   */
  std::size_t operator()(dd::ComplexValue const& c) const noexcept;
};
} // namespace std
