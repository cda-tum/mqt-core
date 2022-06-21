#pragma once

#include <stdexcept>
#include <string>

namespace zx {
    enum class EdgeType { Simple,
                          Hadamard };
    enum class VertexType { Boundary,
                            Z,
                            X };
    using Vertex = std::size_t;
    using Col    = int;
    using Qubit  = int;
    using fp     = double;

    constexpr double        MAX_DENOM           = 1e9; // TODO: maybe too high
    constexpr double        PARAMETER_TOLERANCE = 1e-13;
    constexpr double        TOLERANCE           = 1e-13;
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
