#ifndef JKQZX_INCLUDE_DEFINITIONS_HPP_
#define JKQZX_INCLUDE_DEFINITIONS_HPP_

#include <stdexcept>
namespace zx {
constexpr long MAX_DENOM = 1e17; // TODO: maybe too high
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
