#include "python/pybind11.hpp"

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

// forward declarations
void registerPermutation(py::module& m);
void registerOperations(py::module& m);
void registerSymbolic(py::module& m);
void registerQuantumComputation(py::module& m);

PYBIND11_MODULE(_core, m) {
  registerPermutation(m);

  py::module symbolic = m.def_submodule("symbolic");
  registerSymbolic(symbolic);

  py::module operations = m.def_submodule("operations");
  registerOperations(operations);

  registerQuantumComputation(m);
}

} // namespace mqt
