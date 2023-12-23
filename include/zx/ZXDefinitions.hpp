#pragma once

#include "operations/Expression.hpp"
#include "zx/Rational.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace zx {

enum class EdgeType : uint8_t { Simple, Hadamard };
inline std::ostream& operator<<(std::ostream& os, const EdgeType& type) {
  switch (type) {
  case EdgeType::Simple:
    os << "Simple";
    break;
  case EdgeType::Hadamard:
    os << "Hadamard";
    break;
  }
  return os;
}

enum class VertexType : uint8_t { Boundary, Z, X };
inline std::ostream& operator<<(std::ostream& os, const VertexType& type) {
  switch (type) {
  case VertexType::Boundary:
    os << "Boundary";
    break;
  case VertexType::Z:
    os << "Z";
    break;
  case VertexType::X:
    os << "X";
    break;
  }
  return os;
}

using Vertex = std::size_t;
using Col = std::int32_t;
using Qubit = std::int32_t;
using fp = double;

constexpr fp MAX_DENOM = 1e9; // TODO: maybe too high
constexpr fp PARAMETER_TOLERANCE = 1e-13;
constexpr fp TOLERANCE = 1e-13;
static constexpr auto PI = static_cast<fp>(
    3.141592653589793238462643383279502884197169399375105820974L);

using PiExpression = sym::Expression<double, PiRational>;

class ZXException : public std::invalid_argument {
  std::string msg;

public:
  explicit ZXException(std::string m)
      : std::invalid_argument("ZX Exception"), msg(std::move(m)) {}

  [[nodiscard]] const char* what() const noexcept override {
    return msg.c_str();
  }
};

using gf2Mat = std::vector<std::vector<bool>>;
using gf2Vec = std::vector<bool>;
} // namespace zx
