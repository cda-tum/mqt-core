#include "python/pybind11.hpp"

namespace mqt {

// forward declarations
void registerVariable(py::module& m);
void registerTerm(py::module& m);
void registerExpression(py::module& m);

void registerSymbolic(pybind11::module& m) {
  registerVariable(m);
  registerTerm(m);
  registerExpression(m);
}
} // namespace mqt
