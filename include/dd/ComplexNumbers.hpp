#pragma once

#include "dd/Complex.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/RealNumber.hpp"
#include "dd/RealNumberUniqueTable.hpp"

namespace dd {
/// A class for managing complex numbers in the DD package.
class ComplexNumbers {

public:
  /// Default constructor.
  ComplexNumbers(RealNumberUniqueTable& table, MemoryManager<RealNumber>& cache)
      : uniqueTable(&table), cacheManager(&cache){};
  /// Default destructor.
  ~ComplexNumbers() = default;

  /**
   * @brief Set the numerical tolerance for comparisons of floats.
   * @param tol The new tolerance.
   */
  static void setTolerance(fp tol) noexcept;

  /**
   * @brief Add two complex numbers.
   * @param r The result.
   * @param a The first operand.
   * @param b The second operand.
   * @note Assumes that the entry pointers of the result are aligned.
   */
  static void add(Complex& r, const Complex& a, const Complex& b) noexcept;

  /**
   * @brief Subtract two complex numbers.
   * @param r The result.
   * @param a The first operand.
   * @param b The second operand.
   * @note Assumes that the entry pointers of the result are aligned.
   */
  static void sub(Complex& r, const Complex& a, const Complex& b) noexcept;

  /**
   * @brief Multiply two complex numbers.
   * @param r The result.
   * @param a The first operand.
   * @param b The second operand.
   * @note Assumes that the entry pointers of the result are aligned.
   */
  static void mul(Complex& r, const Complex& a, const Complex& b) noexcept;

  /**
   * @brief Divide two complex numbers.
   * @param r The result.
   * @param a The first operand.
   * @param b The second operand.
   * @note Assumes that the entry pointers of the result are aligned.
   */
  static void div(Complex& r, const Complex& a, const Complex& b) noexcept;

  /**
   * @brief Compute the squared magnitude of a complex number.
   * @param a The complex number.
   * @returns The squared magnitude.
   */
  [[nodiscard]] static fp mag2(const Complex& a) noexcept;

  /**
   * @brief Compute the magnitude of a complex number.
   * @param a The complex number.
   * @returns The magnitude.
   */
  [[nodiscard]] static fp mag(const Complex& a) noexcept;

  /**
   * @brief Compute the argument of a complex number.
   * @param a The complex number.
   * @returns The argument.
   */
  [[nodiscard]] static fp arg(const Complex& a) noexcept;

  /**
   * @brief Compute the complex conjugate of a complex number.
   * @param a The complex number.
   * @returns The complex conjugate.
   * @note Conjugation is efficiently handled by just flipping the sign of the
   * imaginary pointer.
   */
  [[nodiscard]] static Complex conj(const Complex& a) noexcept;

  /**
   * @brief Compute the negation of a complex number.
   * @param a The complex number.
   * @returns The negation.
   * @note Negation is efficiently handled by just flipping the sign of both
   * pointers.
   */
  [[nodiscard]] static Complex neg(const Complex& a) noexcept;

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
   * @brief Lookup a complex value in the complex table; if not found add it.
   * @param c The complex number.
   * @param cached Used to indicate whether the number to be looked up is from
   * the cache or not. If true, the number is returned to the cache as part of
   * the lookup.
   * @return
   */
  [[nodiscard]] Complex lookup(const Complex& c, bool cached = false);

  /**
   * @see lookup(fp r, fp i)
   */
  [[nodiscard]] Complex lookup(const std::complex<fp>& c);

  /**
   * @see lookup(fp r, fp i)
   */
  [[nodiscard]] Complex lookup(const ComplexValue& c);

  /**
   * @brief Lookup a real value in the complex table; if not found add it.
   * @param r The real number.
   * @return The found or added complex number with real part r and imaginary
   * part zero.
   */
  [[nodiscard]] Complex lookup(fp r);

  /**
   * @brief Lookup a complex value in the complex table; if not found add it.
   * @param r The real part.
   * @param i The imaginary part.
   * @return The found or added complex number.
   * @see ComplexTable::lookup
   */
  [[nodiscard]] Complex lookup(fp r, fp i);

  /**
   * @brief Increment the reference count of a complex number.
   * @details This is a pass-through function that increments the reference
   * count of the real and imaginary parts of the given complex number.
   * @param c The complex number
   * @see RealNumberUniqueTable::incRef(RealNumber*)
   */
  void incRef(const Complex& c) const noexcept;

  /**
   * @brief Decrement the reference count of a complex number.
   * @details This is a pass-through function that decrements the reference
   * count of the real and imaginary parts of the given complex number.
   * @param c The complex number
   * @see RealNumberUniqueTable::decRef(RealNumber*)
   */
  void decRef(const Complex& c) const noexcept;

  /**
   * @brief Check whether a complex number is one of the static ones.
   * @param c The complex number.
   * @return Whether the complex number is one of the static ones.
   */
  [[nodiscard]] static constexpr bool isStaticComplex(const Complex& c) {
    return c.exactlyZero() || c.exactlyOne();
  }

  /**
   * @brief Get a complex number from the complex cache.
   * @param c The complex number.
   * @return The cached complex number.
   * @see MemoryManager::get
   */
  [[nodiscard]] Complex getCached();

  /**
   * @brief Get a complex number from the complex cache.
   * @param r The real part.
   * @param i The imaginary part.
   * @return The cached complex number.
   * @see MemoryManager::get
   */
  [[nodiscard]] Complex getCached(fp r, fp i);

  /**
   * @brief Get a complex number from the complex cache.
   * @param c The complex value.
   * @return The cached complex number.
   * @see MemoryManager::get
   */
  [[nodiscard]] Complex getCached(const ComplexValue& c);

  /**
   * @brief Get a complex number from the complex cache.
   * @param c The complex number.
   * @return The cached complex number.
   * @see MemoryManager::get
   */
  [[nodiscard]] Complex getCached(const std::complex<fp>& c);

  /**
   * @brief Get a complex number from the complex cache.
   * @param c The complex number.
   * @return The cached complex number.
   * @see MemoryManager::get
   */
  [[nodiscard]] Complex getCached(const Complex& c);

  /**
   * @brief Return a complex number to the complex cache.
   * @param c The complex number.
   * @see MemoryManager::free
   * @note This method takes care that it never returns a static complex number
   * to the cache. This means it can be called with any complex number. The
   * real and imaginary parts are returned in reverse order to improve cache
   * locality.
   */
  void returnToCache(const Complex& c) noexcept;

  /**
   * @brief Get the number of cached numbers.
   * @return The number of cached numbers.
   */
  [[nodiscard]] std::size_t cacheCount() const noexcept;

  /**
   * @brief Get the number of stored real numbers.
   * @return The number of stored real numbers.
   */
  [[nodiscard]] std::size_t realCount() const noexcept;

private:
  /// A pointer to the unique table to use for calculations
  RealNumberUniqueTable* uniqueTable;
  /// A pointer to the cache manager to use for calculations
  MemoryManager<RealNumber>* cacheManager;
};
} // namespace dd
