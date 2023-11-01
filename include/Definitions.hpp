#pragma once

#include <bitset>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
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

using BitString = std::bitset<128>;

// floating-point type used throughout the library
using fp = double;

constexpr fp PARAMETER_TOLERANCE = 1e-13;

static constexpr fp PI = static_cast<fp>(
    3.141592653589793238462643383279502884197169399375105820974L);
static constexpr fp PI_2 = static_cast<fp>(
    1.570796326794896619231321691639751442098584699687552910487L);
static constexpr fp PI_4 = static_cast<fp>(
    0.785398163397448309615660845819875721049292349843776455243L);

// forward declaration
class Operation;

// supported file formats
enum class Format { Real, OpenQASM, GRCS, TFC, QC, Tensor };

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
} // namespace qc
