#pragma once

#include "dd/DDDefinitions.hpp"

namespace dd {
/**
 * @brief A struct for representing real numbers as part of the DD package.
 * @details Consists of a floating point number (the value), a next pointer
 * (used for chaining entries), and a reference count.
 * @note Due to the way the sign of the value is encoded, special care has to
 * be taken when accessing the value. The static functions in this struct
 * provide safe access to the value of a RealNumber* pointer.
 */
struct RealNumber {

  /**
   * @brief Check whether the number points to the zero number.
   * @param e The number to check.
   * @returns Whether the number points to zero.
   */
  [[nodiscard]] static bool exactlyZero(const RealNumber* e) noexcept;

  /**
   * @brief Check whether the number points to the one number.
   * @param e The number to check.
   * @returns Whether the number points to one.
   */
  [[nodiscard]] static bool exactlyOne(const RealNumber* e) noexcept;

  /**
   * @brief Check whether the number points to the sqrt(2)/2 = 1/sqrt(2) number.
   * @param e The number to check.
   * @returns Whether the number points to negative one.
   */
  [[nodiscard]] static bool exactlySqrt2over2(const RealNumber* e) noexcept;

  /**
   * @brief Get the value of the number.
   * @param e The number to get the value for.
   * @returns The value of the number.
   * @note This function accounts for the sign of the number embedded in the
   * memory address of the number.
   */
  [[nodiscard]] static fp val(const RealNumber* e) noexcept;

  /**
   * @brief Get the reference count of the number.
   * @param num A pointer to the number to get the reference count for.
   * @returns The reference count of the number.
   * @note This function accounts for the sign of the number embedded in the
   * memory address of the number.
   */
  [[nodiscard]] static RefCount refCount(const RealNumber* num) noexcept;

  /**
   * @brief Check whether two floating point numbers are approximately equal.
   * @details This function checks whether two floating point numbers are
   * approximately equal. The two numbers are considered approximately equal
   * if the absolute difference between them is smaller than a small value
   * (TOLERANCE). This function is used to compare floating point numbers
   * stored in the table.
   * @param left The first floating point number.
   * @param right The second floating point number.
   * @returns Whether the two floating point numbers are approximately equal.
   */
  [[nodiscard]] static bool approximatelyEquals(fp left, fp right) noexcept;

  /**
   * @brief Check whether two numbers are approximately equal.
   * @details This function checks whether two numbers are approximately
   * equal. Two numbers are considered approximately equal if they point to
   * the same number or if the values of the numbers are approximately equal.
   * @param left The first number.
   * @param right The second number.
   * @returns Whether the two numbers are approximately equal.
   * @see approximatelyEquals(fp, fp)
   */
  [[nodiscard]] static bool
  approximatelyEquals(const RealNumber* left, const RealNumber* right) noexcept;

  /**
   * @brief Check whether a floating point number is approximately zero.
   * @param e The floating point number to check.
   * @returns Whether the floating point number is approximately zero.
   */
  [[nodiscard]] static bool approximatelyZero(fp e) noexcept;

  /**
   * @brief Check whether a number is approximately zero.
   * @param e The number to check.
   * @returns Whether the number is approximately zero.
   * @see approximatelyZero(fp)
   */
  [[nodiscard]] static bool approximatelyZero(const RealNumber* e) noexcept;

  /**
   * @brief Check whether a floating point number is approximately one.
   * @param e The floating point number to check.
   * @returns Whether the floating point number is approximately one.
   */
  [[nodiscard]] static bool approximatelyOne(fp e) noexcept;

  /**
   * @brief Check whether a number is approximately one.
   * @param e The number to check.
   * @returns Whether the number is approximately one.
   * @see approximatelyOne(fp)
   */
  [[nodiscard]] static bool approximatelyOne(const RealNumber* e) noexcept;

  /**
   * @brief Indicates whether a given number needs reference count updates.
   * @details This function checks whether a given number needs reference count
   * updates. A number needs reference count updates if the pointer to it is
   * not the null pointer, if it is not one of the special numbers (zero,
   * one, 1/sqrt(2)), and if the reference count has saturated.
   * @param num Pointer to the number to check.
   * @returns Whether the number needs reference count updates.
   * @note This function assumes that the pointer to the number is aligned.
   */
  [[nodiscard]] static bool noRefCountingNeeded(const RealNumber* num) noexcept;

  /**
   * @brief Increment the reference count of a number.
   * @details This function increments the reference count of a number. If the
   * reference count has saturated (i.e. reached the maximum value of RefCount)
   * the reference count is not incremented.
   * @param num A pointer to the number to increment the reference count of.
   * @returns Whether the reference count was incremented.
   * @note Typically, you do not want to call this function directly. Instead,
   * use the RealNumberUniqueTable::incRef(RelNumber*) function.
   */
  [[nodiscard]] static bool incRef(const RealNumber* num) noexcept;

  /**
   * @brief Decrement the reference count of a number.
   * @details This function decrements the reference count of a number. If the
   * reference count has saturated (i.e. reached the maximum value of RefCount)
   * the reference count is not decremented.
   * @param num A pointer to the number to decrement the reference count of.
   * @returns Whether the reference count was decremented.
   * @note Typically, you do not want to call this function directly. Instead,
   * use the RealNumberUniqueTable::decRef(RelNumber*) function.
   */
  [[nodiscard]] static bool decRef(const RealNumber* num) noexcept;

  /**
   * @brief Write a binary representation of the number to a stream.
   * @param e The number to write.
   * @param os The stream to write to.
   */
  static void writeBinary(const RealNumber* e, std::ostream& os);

  /**
   * @brief Get an aligned pointer to the number.
   * @details Since the least significant bit of the memory address of the
   * number is used to encode the sign of the value, the pointer to the number
   * might not be aligned. This function returns an aligned pointer to the
   * number.
   * @param e The number to get the aligned pointer for.
   * @returns An aligned pointer to the number.
   */
  [[nodiscard]] static RealNumber*
  getAlignedPointer(const RealNumber* e) noexcept;

  /**
   * @brief Get a pointer to the number with a negative sign.
   * @details Since the least significant bit of the memory address of the
   * number is used to encode the sign of the value, this function just sets
   * the least significant bit of the memory address of the number to 1.
   * @param e The number to get the negative pointer for.
   * @returns A negative pointer to the number.
   */
  [[nodiscard]] static RealNumber*
  getNegativePointer(const RealNumber* e) noexcept;

  /**
   * @brief Check whether the number is a negative pointer.
   * @param e The number to check.
   * @returns Whether the number is a negative pointer.
   */
  [[nodiscard]] static bool isNegativePointer(const RealNumber* e) noexcept;

  /**
   * @brief Flip the sign of the number pointer.
   * @param e The number to flip the sign of.
   * @returns The number with the sign flipped.
   * @note This function does not change the sign of the value of the number.
   * It rather changes the sign of the pointer to the number.
   * @note We do not consider negative zero here, since it is not used in the
   * DD package. There only exists one zero number, which is positive.
   */
  [[nodiscard]] static RealNumber*
  flipPointerSign(const RealNumber* e) noexcept;

  /**
   * @brief The value of the number.
   * @details The value of the number is a floating point number. The sign of
   * the value is encoded in the least significant bit of the memory address
   * of the number. As a consequence, values stored here are always
   * non-negative. The sign of the value as well as the value itself can be
   * accessed using the static functions of this struct.
   */
  fp value{};
  /**
   * @brief The next pointer of the number.
   * @details The next pointer is used to from linked lists.The next pointer
   * is part of this class for efficiency reasons. It could be stored
   * separately, but that would require many small allocations. This way, the
   * entries can be allocated in chunks, which is much more efficient.
   */
  RealNumber* next{};
  /**
   * @brief The reference count of the number.
   * @details The reference count is used to determine whether a number is
   * still in use. If the reference count is zero, the number is not in use
   * and can be garbage collected.
   */
  RefCount ref{};

  /// numerical tolerance to be used for floating point values
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static inline fp eps = std::numeric_limits<dd::fp>::epsilon() * 1024;
};

namespace constants {
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
/// The constant zero.
extern RealNumber zero;
/// The constant one.
extern RealNumber one;
/// The constant sqrt(2)/2 = 1/sqrt(2).
extern RealNumber sqrt2over2;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/**
 * @brief Check whether a number is one of the static numbers.
 * @param e The number to check.
 * @return Whether the number is one of the static numbers.
 */
[[nodiscard]] constexpr bool isStaticNumber(const RealNumber* e) noexcept {
  return e == &zero || e == &one || e == &sqrt2over2;
}
} // namespace constants
} // namespace dd
