#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qc {
class QFRException : public std::invalid_argument {
  std::string msg;

public:
  explicit QFRException(std::string m)
      : std::invalid_argument("QFR Exception"), msg(std::move(m)) {}

  [[nodiscard]] const char* what() const noexcept override {
    return msg.c_str();
  }
};

using Qubit = std::uint32_t;
using Bit = std::uint64_t;

template <class IdxType, class SizeType>
using Register = std::pair<IdxType, SizeType>;
using QuantumRegister = Register<Qubit, std::size_t>;
using ClassicalRegister = Register<Bit, std::size_t>;
template <class RegisterType>
using RegisterMap = std::map<std::string, RegisterType, std::greater<>>;
using QuantumRegisterMap = RegisterMap<QuantumRegister>;
using ClassicalRegisterMap = RegisterMap<ClassicalRegister>;
using RegisterNames = std::vector<std::pair<std::string, std::string>>;

using Targets = std::vector<Qubit>;

using BitString = std::bitset<4096>;

// floating-point type used throughout the library
using fp = double;

constexpr fp PARAMETER_TOLERANCE = 1e-13;

static constexpr fp PI = static_cast<fp>(
    3.141592653589793238462643383279502884197169399375105820974L);
static constexpr fp PI_2 = static_cast<fp>(
    1.570796326794896619231321691639751442098584699687552910487L);
static constexpr fp PI_4 = static_cast<fp>(
    0.785398163397448309615660845819875721049292349843776455243L);
static constexpr fp TAU = static_cast<fp>(
    6.283185307179586476925286766559005768394338798750211641950L);
static constexpr fp E = static_cast<fp>(
    2.718281828459045235360287471352662497757247093699959574967L);

static constexpr size_t OUTPUT_INDENT_SIZE = 2;

// forward declaration
class Operation;

// supported file formats
enum class Format : uint8_t {
  Real,
  OpenQASM2,
  OpenQASM3,
  GRCS,
  TFC,
  QC,
  Tensor
};

using DAG = std::vector<std::deque<std::unique_ptr<Operation>*>>;
using DAGIterator = std::deque<std::unique_ptr<Operation>*>::iterator;
using DAGReverseIterator =
    std::deque<std::unique_ptr<Operation>*>::reverse_iterator;
using DAGIterators = std::vector<DAGIterator>;
using DAGReverseIterators = std::vector<DAGReverseIterator>;

/**
 * @brief 64bit mixing hash (from MurmurHash3)
 * @details Hash function for 64bit integers adapted from MurmurHash3
 * @param k the number to hash
 * @returns the hash value
 * @see https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 */
[[nodiscard]] constexpr std::size_t murmur64(std::size_t k) noexcept {
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
[[nodiscard]] constexpr std::size_t
combineHash(const std::size_t lhs, const std::size_t rhs) noexcept {
  return lhs ^ (rhs + 0x9e3779b97f4a7c15ULL + (lhs << 6) + (lhs >> 2));
}

/**
 * @brief Extend a 64bit hash with a 64bit integer
 * @param hash The hash to extend
 * @param with The integer to extend the hash with
 * @return The combined hash
 */
constexpr void hashCombine(std::size_t& hash, const std::size_t with) noexcept {
  hash = combineHash(hash, with);
}

/// Pairs do not provide a hash function by default, this is the replacement
template <class T> struct PairHash {
  size_t operator()(const std::pair<T, T>& x) const {
    return combineHash(std::hash<T>{}(x.first), std::hash<T>{}(x.second));
  }
};

/**
 * @brief Function used to mark unreachable code
 * @details Uses compiler specific extensions if possible. Even if no extension
 * is used, undefined behavior is still raised by an empty function body and the
 * noreturn attribute.
 */
[[noreturn]] inline void unreachable() {
#ifdef __GNUC__ // GCC, Clang, ICC
  __builtin_unreachable();
#elif defined(_MSC_VER) // MSVC
  __assume(false);
#endif
}
} // namespace qc
