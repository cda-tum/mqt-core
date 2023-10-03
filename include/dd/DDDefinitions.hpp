#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dd {
/**
 * @brief Integer type used for indexing qubits
 * @details `std::uint16_t` can address up to 65536 qubits as [0, ..., 65535].
 * @note If you need even more qubits, this can be increased to `std::uint32_t`.
 * Beware of the increased memory footprint of matrix nodes.
 */
using Qubit = std::uint16_t;

/**
 * @brief Integer type used for reference counting
 * @details Allows a maximum reference count of roughly 4 billion.
 */
using RefCount = std::uint32_t;
static_assert(std::is_unsigned_v<RefCount>, "RefCount should be unsigned.");

/**
 * @brief Floating point type to use for computations
 * @note Adjusting the precision might lead to unexpected results.
 */
using fp = double;
static_assert(
    std::is_floating_point_v<fp>,
    "fp should be a floating point type (float, *double*, long double)");

// logic radix
static constexpr std::uint8_t RADIX = 2;
// max no. of edges = RADIX^2
static constexpr std::uint8_t NEDGE = RADIX * RADIX;

enum class BasisStates {
  zero,  // NOLINT(readability-identifier-naming)
  one,   // NOLINT(readability-identifier-naming)
  plus,  // NOLINT(readability-identifier-naming)
  minus, // NOLINT(readability-identifier-naming)
  right, // NOLINT(readability-identifier-naming)
  left   // NOLINT(readability-identifier-naming)
};

static constexpr fp SQRT2_2 = static_cast<fp>(
    0.707106781186547524400844362104849039284835937688474036588L);
static constexpr fp PI = static_cast<fp>(
    3.141592653589793238462643383279502884197169399375105820974L);
static constexpr fp PI_2 = static_cast<fp>(
    1.570796326794896619231321691639751442098584699687552910487L);
static constexpr fp PI_4 = static_cast<fp>(
    0.785398163397448309615660845819875721049292349843776455243L);

static constexpr std::uint64_t SERIALIZATION_VERSION = 1;

/**
 * @brief 64bit mixing hash (from MurmurHash3)
 * @details Hash function for 64bit integers adapted from MurmurHash3
 * @param k the number to hash
 * @returns the hash value
 * @see https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 */
constexpr std::size_t murmur64(std::size_t k) noexcept {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

/**
 * @brief Combine two 64bit hashes into one 64bit hash
 * @details Combines two 64bit hashes into one 64bit hash based on
 * boost::hash_combine (https://www.boost.org/LICENSE_1_0.txt)
 * @param lhs The first hash
 * @param rhs The second hash
 * @returns The combined hash
 */
constexpr std::size_t combineHash(std::size_t lhs, std::size_t rhs) noexcept {
  lhs ^= rhs + 0x9e3779b97f4a7c15ULL + (lhs << 6) + (lhs >> 2);
  return lhs;
}

struct PairHash {
  std::size_t
  operator()(const std::pair<std::size_t, std::size_t>& p) const noexcept {
    return combineHash(p.first, p.second);
  }
};

using CVec = std::vector<std::complex<fp>>;
using SparseCVec = std::unordered_map<std::size_t, std::complex<fp>>;
using SparsePVec = std::unordered_map<std::size_t, fp>;
using SparsePVecStrKeys = std::unordered_map<std::string, fp>;
using CMat = std::vector<CVec>;
using SparseCMat = std::unordered_map<std::pair<std::size_t, std::size_t>,
                                      std::complex<fp>, PairHash>;

/**
 * @brief Converts a decimal number to a binary string (big endian)
 * @param value The decimal number to convert
 * @param nbits The number of bits to use for the binary representation
 * @return The binary representation of the decimal number
 */
[[nodiscard]] static inline std::string
intToBinaryString(const std::size_t value, const std::size_t nbits) {
  std::string binary(nbits, '0');
  for (std::size_t j = 0; j < nbits; ++j) {
    if ((value & (1U << j)) != 0U) {
      binary[nbits - 1 - j] = '1';
    }
  }
  return binary;
}

// calculates the Units in Last Place (ULP) distance of two floating point
// numbers
[[maybe_unused]] static std::size_t ulpDistance(fp a, fp b) {
  // NOLINTNEXTLINE(clang-diagnostic-float-equal)
  if (a == b) {
    return 0;
  }

  std::size_t ulps = 1;
  fp nextFP = std::nextafter(a, b);
  // NOLINTNEXTLINE(clang-diagnostic-float-equal)
  while (nextFP != b) {
    ulps++;
    nextFP = std::nextafter(nextFP, b);
  }
  return ulps;
}
} // namespace dd
