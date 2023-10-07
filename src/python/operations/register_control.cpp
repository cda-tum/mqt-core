#include "operations/Control.hpp"
#include "python/pybind11.hpp"

namespace mqt {

void registerControl(py::module& m) {

  auto control = py::class_<qc::Control>(m, "Control");
  auto controlType = py::enum_<qc::Control::Type>(control, "Type");
  controlType.value("Pos", qc::Control::Type::Pos);
  controlType.value("Neg", qc::Control::Type::Neg);
  controlType.def(
      "__str__",
      [](const qc::Control::Type& type) {
        return type == qc::Control::Type::Pos ? "Pos" : "Neg";
      },
      py::prepend());
  controlType.def(
      "__repr__",
      [](const qc::Control::Type& type) {
        return type == qc::Control::Type::Pos ? "Pos" : "Neg";
      },
      py::prepend());
  controlType.def("__bool__", [](const qc::Control::Type& type) {
    return type == qc::Control::Type::Pos;
  });
  py::implicitly_convertible<py::bool_, qc::Control::Type>();
  py::implicitly_convertible<py::str, qc::Control::Type>();

  control.def(py::init<qc::Qubit, qc::Control::Type>(), "qubit"_a,
              "type_"_a = qc::Control::Type::Pos,
              "Create a control qubit of the specified control type.");
  control.def_readwrite(
      "type_", &qc::Control::type,
      "The type of the control qubit. Can be positive or negative.");
  control.def_readwrite("qubit", &qc::Control::qubit,
                        "The qubit index of the control qubit.");
  control.def("__str__", [](const qc::Control& c) { return c.toString(); });
  control.def("__repr__", [](const qc::Control& c) { return c.toString(); });
  py::implicitly_convertible<py::int_, qc::Control>();
}

} // namespace mqt
