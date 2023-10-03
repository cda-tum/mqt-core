#include "operations/Control.hpp"
#include "python/pybind11.hpp"

namespace mqt {

void registerControl(py::module& m) {

  auto control = py::class_<qc::Control>(m, "Control");
  control
      .def(py::init<qc::Qubit, qc::Control::Type>(), "qubit"_a,
           "type_"_a = qc::Control::Type::Pos,
           "Create a control qubit of the specified control type.")
      .def_readwrite(
          "type_", &qc::Control::type,
          "The type of the control qubit. Can be positive or negative.")
      .def_readwrite("qubit", &qc::Control::qubit,
                     "The qubit index of the control qubit.");

  py::enum_<qc::Control::Type>(control, "Type")
      .value("Pos", qc::Control::Type::Pos)
      .value("Neg", qc::Control::Type::Neg)
      .export_values();
  py::implicitly_convertible<py::str, qc::Control::Type>();
}

} // namespace mqt
