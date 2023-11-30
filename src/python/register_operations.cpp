#include "python/pybind11.hpp"

namespace mqt {

// forward declarations
void registerOptype(py::module& m);
void registerControl(py::module& m);
void registerOperation(py::module& m);
void registerStandardOperation(py::module& m);
void registerCompoundOperation(py::module& m);
void registerNonUnitaryOperation(py::module& m);
void registerSymbolicOperation(py::module& m);
void registerClassicControlledOperation(py::module& m);

void registerOperations(py::module& m) {
  registerOptype(m);
  registerControl(m);
  registerOperation(m);
  registerStandardOperation(m);
  registerCompoundOperation(m);
  registerNonUnitaryOperation(m);
  registerSymbolicOperation(m);
  registerClassicControlledOperation(m);
}
} // namespace mqt
