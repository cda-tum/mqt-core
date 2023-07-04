#pragma once

#include "dd/ComplexCache.hpp"
#include "dd/ComplexTable.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/Definitions.hpp"

namespace dd {
/// A class for managing complex numbers in the DD package.
struct ComplexNumbers {
  /// The hash table for complex numbers.
  ComplexTable complexTable{};
  /// The cache for complex numbers.
  ComplexCache complexCache{};

  /// Default constructor.
  ComplexNumbers() = default;
  /// Default destructor.
  ~ComplexNumbers() = default;

  /**
   * @brief Clear both the hash table and the cache.
   * @see ComplexTable::clear
   * @see ComplexCache::clear
   */
  void clear();

  /**
   * @brief Set the numerical tolerance for complex numbers.
   * @param tol The new tolerance.
   * @see ComplexTable::setTolerance
   */
  static void setTolerance(fp tol);

  /**
   * @brief Add two complex numbers.
   * @param r The result.
   * @param a The first operand.
   * @param b The second operand.
   * @note Assumes that the entry pointers of the result are aligned.
   */
  static void add(Complex& r, const Complex& a, const Complex& b);

  /**
   * @brief Subtract two complex numbers.
   * @param r The result.
   * @param a The first operand.
   * @param b The second operand.
   * @note Assumes that the entry pointers of the result are aligned.
   */
  static void sub(Complex& r, const Complex& a, const Complex& b);

  /**
   * @brief Multiply two complex numbers.
   * @param r The result.
   * @param a The first operand.
   * @param b The second operand.
   * @note Assumes that the entry pointers of the result are aligned.
   */
  static void mul(Complex& r, const Complex& a, const Complex& b);

  /**
   * @brief Divide two complex numbers.
   * @param r The result.
   * @param a The first operand.
   * @param b The second operand.
   * @note Assumes that the entry pointers of the result are aligned.
   */
  static void div(Complex& r, const Complex& a, const Complex& b);

  /**
   * @brief Compute the squared magnitude of a complex number.
   * @param a The complex number.
   * @returns The squared magnitude.
   */
  [[nodiscard]] static fp mag2(const Complex& a);

  /**
   * @brief Compute the magnitude of a complex number.
   * @param a The complex number.
   * @returns The magnitude.
   */
  [[nodiscard]] static fp mag(const Complex& a);

  /**
   * @brief Compute the argument of a complex number.
   * @param a The complex number.
   * @returns The argument.
   */
  [[nodiscard]] static fp arg(const Complex& a);

  /**
   * @brief Compute the complex conjugate of a complex number.
   * @param a The complex number.
   * @returns The complex conjugate.
   * @note Conjugation is efficiently handled by just flipping the sign of the
   * imaginary pointer.
   */
  [[nodiscard]] static Complex conj(const Complex& a);

  /**
   * @brief Compute the negation of a complex number.
   * @param a The complex number.
   * @returns The negation.
   * @note Negation is efficiently handled by just flipping the sign of both
   * pointers.
   */
  [[nodiscard]] static Complex neg(const Complex& a);

  /**
   * @brief Add two complex numbers and return the result in a new complex
   * number taken from the cache.
   * @param a The first operand.
   * @param b The second operand.
   * @return The result.
   */
  [[nodiscard]] Complex addCached(const Complex& a, const Complex& b);

  /**
   * @brief Subtract two complex numbers and return the result in a new complex
   * number taken from the cache.
   * @param a The first operand.
   * @param b The second operand.
   * @return The result.
   */
  [[nodiscard]] Complex subCached(const Complex& a, const Complex& b);

  /**
   * @brief Multiply two complex numbers and return the result in a new complex
   * number taken from the cache.
   * @param a The first operand.
   * @param b The second operand.
   * @return The result.
   */
  [[nodiscard]] Complex mulCached(const Complex& a, const Complex& b);

  /**
   * @brief Divide two complex numbers and return the result in a new complex
   * number taken from the cache.
   * @param a The first operand.
   * @param b The second operand.
   * @return The result.
   */
  [[nodiscard]] Complex divCached(const Complex& a, const Complex& b);

  /**
   * @see lookup(fp r, fp i)
   */
  [[nodiscard]] Complex lookup(const Complex& c);

  /**
   * @see lookup(fp r, fp i)
   */
  [[nodiscard]] Complex lookup(const std::complex<fp>& c);

  /**
   * @see lookup(fp r, fp i)
   */
  [[nodiscard]] Complex lookup(const ComplexValue& c);

  /**
   * @brief Lookup a complex value in the complex table; if not found add it.
   * @param r The real part.
   * @param i The imaginary part.
   * @return The found or added complex number.
   * @see ComplexTable::lookup
   */
  [[nodiscard]] Complex lookup(fp r, fp i);

  /**
   * @brief Check whether a complex number is one of the static ones.
   * @param c The complex number.
   * @return Whether the complex number is one of the static ones.
   */
  [[nodiscard]] static bool isStaticComplex(const Complex& c) {
    return &c == &Complex::zero || &c == &Complex::one;
  }

  /**
   * @brief Increment the reference count of a complex number.
   * @param c The complex number.
   * @see ComplexTable::incRef
   */
  static void incRef(const Complex& c);

  /**
   * @brief Decrement the reference count of a complex number.
   * @param c The complex number.
   * @see ComplexTable::decRef
   */
  static void decRef(const Complex& c);

  /**
   * @brief Garbage collect the complex table.
   * @param force Whether to force garbage collection.
   * @return The number of collected entries.
   * @see ComplexTable::garbageCollect
   */
  std::size_t garbageCollect(bool force = false);

  /**
   * @brief Get a temporary complex number from the complex cache.
   * @return The temporary complex number.
   * @see ComplexCache::getTemporaryComplex
   */
  [[nodiscard]] Complex getTemporary();

  /**
   * @brief Get a temporary complex number from the complex cache.
   * @param r The real part.
   * @param i The imaginary part.
   * @return The temporary complex number.
   * @see ComplexCache::getTemporaryComplex
   */
  [[nodiscard]] Complex getTemporary(const fp& r, const fp& i);

  /**
   * @brief Get a temporary complex number from the complex cache.
   * @param c The complex value.
   * @return The temporary complex number.
   * @see ComplexCache::getTemporaryComplex
   */
  [[nodiscard]] Complex getTemporary(const ComplexValue& c);

  /**
   * @brief Get a complex number from the complex cache.
   * @param c The complex number.
   * @return The cached complex number.
   * @see ComplexCache::getCachedComplex
   */
  [[nodiscard]] Complex getCached();

  /**
   * @brief Get a complex number from the complex cache.
   * @param r The real part.
   * @param i The imaginary part.
   * @return The cached complex number.
   * @see ComplexCache::getCachedComplex
   */
  [[nodiscard]] Complex getCached(const fp& r, const fp& i);

  /**
   * @brief Get a complex number from the complex cache.
   * @param c The complex value.
   * @return The cached complex number.
   * @see ComplexCache::getCachedComplex
   */
  [[nodiscard]] Complex getCached(const ComplexValue& c);

  /**
   * @brief Get a complex number from the complex cache.
   * @param c The complex number.
   * @return The cached complex number.
   * @see ComplexCache::getCachedComplex
   */
  [[nodiscard]] Complex getCached(const std::complex<fp>& c);

  /**
   * @brief Return a complex number to the complex cache.
   * @param c The complex number.
   * @see ComplexCache::returnToCache
   */
  void returnToCache(Complex& c);

  /**
   * @brief Get the number of complex numbers in the complex cache.
   * @return The number of complex numbers in the complex cache.
   */
  [[nodiscard]] std::size_t cacheCount() const;
};
} // namespace dd
