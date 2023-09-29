#include <python/pybind11.hpp>

namespace mqt {

void registerDDModule(py::module_& m);

PYBIND11_MODULE(_core, m) {
  m.def(
      "add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);

  auto dd = m.def_submodule("dd", "Quantum decision diagram (DD) library");
  registerDDModule(dd);
}

} // namespace mqt
