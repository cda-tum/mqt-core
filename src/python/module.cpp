#include <python/pybind11.hpp>

namespace mqt {

PYBIND11_MODULE(_core, m) {
  m.def(
      "add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
}

} // namespace mqt
