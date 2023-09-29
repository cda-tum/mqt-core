#include "Definitions.hpp"
#include "Permutation.hpp"
#include "QuantumComputation.hpp"
#include "operations/CompoundOperation.hpp"
#include "operations/Control.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"
#include "operations/StandardOperation.hpp"
#include "operations/SymbolicOperation.hpp"

#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <pybind11/stl.h>
#include <python/pybind11.hpp>
#include <sstream>
#include <string>
#include <vector>


namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

  //forward declarations
  void registerOperations(py::module& m);
  void registerSymbolic(py::module& m);
  void registerQuantumComputation(py::module& m);

PYBIND11_MODULE(_core, m) {

  py::module operations = m.def_submodule("operations");
  registerOperations(operations);
  
  py::module symbolic = m.def_submodule("symbolic");
  registerSymbolic(symbolic);

  py::module quantumComputation = m.def_submodule("quantum_computation");
  registerQuantumComputation(quantumComputation);
}

} // namespace mqt
