#ifndef JKQZX_INCLUDE_DEFINITIONS_HPP_
#define JKQZX_INCLUDE_DEFINITIONS_HPP_

#include <stdexcept>
#include "dd/Definitions.hpp"
#include "Rational.hpp"

namespace zx {
  enum class EdgeType { Simple, Hadamard };
enum class VertexType { Boundary, Z, X };
using Vertex = int32_t;
using Col = int32_t;

struct Edge {
  int32_t to;
  EdgeType type;

  Edge()=default;
  Edge(int32_t to, EdgeType type):to(to), type(type) {};
  void toggle() {
    type = (type == EdgeType::Simple) ? EdgeType::Hadamard : EdgeType::Simple;
  }
};

struct VertexData {
  Col col;
  dd::Qubit qubit;
  Rational phase;
  VertexType type;
};
constexpr double MAX_DENOM = 1e9; // TODO: maybe too high
static constexpr double PI =
    3.141592653589793238462643383279502884197169399375105820974L;

  class ZXException: public std::invalid_argument {
    std::string msg;

    public:
        explicit ZXException(std::string msg):
            std::invalid_argument("ZX Exception"), msg(std::move(msg)) {}

        [[nodiscard]] const char* what() const noexcept override {
            return msg.c_str();
        }
  };
} // namespace zx
#endif /* JKQZX_INCLUDE_DEFINITIONS_HPP_ */
