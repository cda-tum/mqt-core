#include "na/NAComputation.hpp"
#include "python/pybind11.hpp"

namespace mqt {

void registerNAComputation(py::module& m) {

  auto qc = py::class_<na::NAComputation>(m, "NAComputation");

  ///---------------------------------------------------------------------------
  ///                           \n Constructors \n
  ///---------------------------------------------------------------------------

  qc.def(py::init<>(), "Constructs an empty NAComputation.");

  ///---------------------------------------------------------------------------
  ///                       \n String Serialization \n
  ///---------------------------------------------------------------------------
  ///
  qc.def("__str__",
         [](const na::NAComputation& circ) { return circ.toString(); });
}

PYBIND11_MODULE(na, m, py::mod_gil_not_used()) { registerNAComputation(m); }

} // namespace mqt
