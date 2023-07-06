#pragma once

#include "dd/Complex.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/Definitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/RealNumber.hpp"
#include "dd/RealNumberUniqueTable.hpp"

namespace dd {
/// A class for managing complex numbers in the DD package.
class ComplexNumbers {

public:
  /// Default constructor.
  ComplexNumbers() = default;
  /// Default destructor.
  ~ComplexNumbers() = default;

  /**
   * @brief Clear both the hash table and the cache.
   * @see ComplexTable::clear
   * @see MemoryManager::reset
   */
  void clear() noexcept;

  /**
   * @brief Set the numerical tolerance for complex numbers.
   * @param tol The new tolerance.
   * @see ComplexTable::setTolerance
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
   * @brief Add two complex numbers and return the result in temporary complex
   * number taken from the cache.
   * @param a The first operand.
   * @param b The second operand.
   * @return The result.
   * @note The result is only valid until the next call to any function that
   * requests a temporary or cached complex number.
   */
  [[nodiscard]] Complex addTemp(const Complex& a, const Complex& b);

  /**
   * @brief Subtract two complex numbers and return the result in temporary
   * complex number taken from the cache.
   * @param a The first operand.
   * @param b The second operand.
   * @return The result.
   * @note The result is only valid until the next call to any function that
   * requests a temporary or cached complex number.
   */
  [[nodiscard]] Complex subTemp(const Complex& a, const Complex& b);

  /**
   * @brief Multiply two complex numbers and return the result in temporary
   * complex number taken from the cache.
   * @param a The first operand.
   * @param b The second operand.
   * @return The result.
   * @note The result is only valid until the next call to any function that
   * requests a temporary or cached complex number.
   */
  [[nodiscard]] Complex mulTemp(const Complex& a, const Complex& b);

  /**
   * @brief Divide two complex numbers and return the result in temporary
   * complex number taken from the cache.
   * @param a The first operand.
   * @param b The second operand.
   * @return The result.
   * @note The result is only valid until the next call to any function that
   * requests a temporary or cached complex number.
   */
  [[nodiscard]] Complex divTemp(const Complex& a, const Complex& b);

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
  static void incRef(const Complex& c) noexcept;

  /**
   * @brief Decrement the reference count of a complex number.
   * @param c The complex number.
   * @see ComplexTable::decRef
   */
  static void decRef(const Complex& c) noexcept;

  /**
   * @brief Garbage collect the complex table.
   * @param force Whether to force garbage collection.
   * @return The number of collected entries.
   * @see ComplexTable::garbageCollect
   */
  std::size_t garbageCollect(bool force = false) noexcept;

  /**
   * @brief Get a temporary complex number from the complex cache.
   * @return The temporary complex number.
   * @see MemoryManager::getTemporaryPair
   */
  [[nodiscard]] Complex getTemporary();

  /**
   * @brief Get a temporary complex number from the complex cache.
   * @param r The real part.
   * @param i The imaginary part.
   * @return The temporary complex number.
   * @see MemoryManager::getTemporaryPair
   */
  [[nodiscard]] Complex getTemporary(fp r, fp i);

  /**
   * @brief Get a temporary complex number from the complex cache.
   * @param c The complex value.
   * @return The temporary complex number.
   * @see MemoryManager::getTemporaryPair
   */
  [[nodiscard]] Complex getTemporary(const ComplexValue& c);

  /**
   * @brief Get a temporary complex number from the complex cache.
   * @param c The complex number.
   * @return The temporary complex number.
   * @see MemoryManager::getTemporaryPair
   */
  [[nodiscard]] Complex getTemporary(const Complex& c);

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

  /// Get the cache manager
  [[nodiscard]] const auto& getCacheManager() const noexcept {
    return cacheManager;
  }

  /// @see MemoryManager::reset
  void resetCache(const bool resizeToTotal = false) noexcept {
    cacheManager.reset(resizeToTotal);
  }

  /// Get the complex table
  [[nodiscard]] const auto& getComplexTable() const noexcept {
    return complexTable;
  }

  /// Get the memory manager
  [[nodiscard]] const auto& getMemoryManager() const noexcept {
    return memoryManager;
  }
  /// Get a mutual reference to the memory manager
  [[nodiscard]] auto& getMemoryManager() noexcept { return memoryManager; }

private:
  /// The memory manager for complex numbers.
  MemoryManager<RealNumber> memoryManager{};
  /// The hash table for complex numbers.
  RealNumberUniqueTable complexTable{memoryManager};
  /// The cache manager for complex numbers.
  MemoryManager<RealNumber> cacheManager{};
};
} // namespace dd
