#pragma once

#include "dd/Definitions.hpp"

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>

namespace dd {

struct RealNumber;

/// A complex number represented by two pointers to compute table entries.
struct Complex {
  /// Compute table entry for the real part.
  RealNumber* r;
  /// Compute table entry for the imaginary part.
  RealNumber* i;

  // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
  /// The static zero constant.
  static Complex zero;
  /// The static one constant.
  static Complex one;
  // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

  /**
   * @brief Set the value based on the given complex number.
   * @param c The value to set.
   */
  void setVal(const Complex& c) const noexcept;

  /**
   * @brief Increment the reference count of a complex number.
   * @see RealNumber::incRef
   */
  void incRef() const noexcept;

  /**
   * @brief Decrement the reference count of a complex number.
   * @see RealNumber::decRef
   */
  void decRef() const noexcept;

  /**
   * @brief Check whether the complex number is exactly equal to zero.
   * @returns True if the complex number is exactly equal to zero, false
   * otherwise.
   * @see CTEntry::exactlyZero
   */
  [[nodiscard]] bool exactlyZero() const noexcept;

  /**
   * @brief Check whether the complex number is exactly equal to one.
   * @returns True if the complex number is exactly equal to one, false
   * otherwise.
   * @see CTEntry::exactlyOne
   * @see CTEntry::exactlyZero
   */
  [[nodiscard]] bool exactlyOne() const noexcept;

  /**
   * @brief Check whether the complex number is approximately equal to the
   * given complex number.
   * @param c The complex number to compare to.
   * @returns True if the complex number is approximately equal to the given
   * complex number, false otherwise.
   * @see CTEntry::approximatelyEquals
   */
  [[nodiscard]] bool approximatelyEquals(const Complex& c) const noexcept;

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
   * @brief Check for exact equality.
   * @param other The complex number to compare to.
   * @returns True if the complex numbers are exactly equal, false otherwise.
   * @note Boils down to a pointer comparison.
   */
  [[nodiscard]] bool operator==(const Complex& other) const noexcept;

  /// @see operator==
  [[nodiscard]] bool operator!=(const Complex& other) const noexcept;

  /**
   * @brief Convert the complex number to a string.
   * @param formatted Whether to apply special formatting to the numbers.
   * @param precision The precision to use for the numbers.
   * @returns The string representation of the complex number.
   * @see ComplexValue::toString
   */
  [[nodiscard]] std::string toString(bool formatted = true,
                                     int precision = -1) const;

  /**
   * @brief Write the complex number to a binary stream.
   * @param os The output stream to write to.
   * @see CTEntry::writeBinary
   */
  void writeBinary(std::ostream& os) const;
};

/**
 * @brief Print a complex number to a stream.
 * @param os The output stream to write to.
 * @param c The complex number to print.
 * @returns The output stream.
 */
std::ostream& operator<<(std::ostream& os, const Complex& c);

} // namespace dd

namespace std {
/// Hash function for complex numbers.
template <> struct hash<dd::Complex> {
  /**
   * @brief Compute the hash value for a complex number.
   * @details Reinterprets the pointers to the real and imaginary part as
   * integers and computes the hash value for those. Afterwards, the two hash
   * values are combined.
   * @param c The complex number to compute the hash value for.
   * @returns The hash value.
   * @see dd::murmur64
   * @see dd::combineHash
   */
  std::size_t operator()(dd::Complex const& c) const noexcept {
    auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(c.r));
    auto h2 = dd::murmur64(reinterpret_cast<std::size_t>(c.i));
    return dd::combineHash(h1, h2);
  }
};
} // namespace std
