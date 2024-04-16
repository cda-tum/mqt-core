#include "operations/Operation.hpp"
#include "pybind11/operators.h"
#include "python/pybind11.hpp"

namespace mqt {

void registerOperation(py::module& m) {
  py::class_<qc::Operation>(m, "Operation")
      .def_property_readonly("name", &qc::Operation::getName)
      .def_property("type_", &qc::Operation::getType, &qc::Operation::setGate)
      .def_property(
          "targets", [](const qc::Operation& op) { return op.getTargets(); },
          &qc::Operation::setTargets)
      .def_property_readonly("num_targets", &qc::Operation::getNtargets)
      .def_property(
          "controls", [](const qc::Operation& op) { return op.getControls(); },
          &qc::Operation::setControls)
      .def_property_readonly("num_controls", &qc::Operation::getNcontrols)
      .def("add_control", &qc::Operation::addControl, "control"_a)
      .def("add_controls", &qc::Operation::addControls, "controls"_a)
      .def("clear_controls", &qc::Operation::clearControls)
      .def(
          "remove_control",
          [](qc::Operation& op, const qc::Control& c) { op.removeControl(c); },
          "control"_a)
      .def("remove_controls", &qc::Operation::removeControls, "controls"_a)
      .def("get_used_qubits", &qc::Operation::getUsedQubits)
      .def("acts_on", &qc::Operation::actsOn, "qubit"_a)
      .def_property(
          "parameter",
          [](const qc::Operation& op) { return op.getParameter(); },
          &qc::Operation::setParameter)
      .def("is_unitary", &qc::Operation::isUnitary)
      .def("is_standard_operation", &qc::Operation::isStandardOperation)
      .def("is_compound_operation", &qc::Operation::isCompoundOperation)
      .def("is_non_unitary_operation", &qc::Operation::isNonUnitaryOperation)
      .def("is_classic_controlled_operation",
           &qc::Operation::isClassicControlledOperation)
      .def("is_symbolic_operation", &qc::Operation::isSymbolicOperation)
      .def("is_controlled", &qc::Operation::isControlled)
      .def("get_inverted", &qc::Operation::getInverted)
      .def("invert", &qc::Operation::invert)
      .def("__eq__", [](const qc::Operation& op,
                        const qc::Operation& other) { return op == other; })
      .def("__ne__", [](const qc::Operation& op,
                        const qc::Operation& other) { return op != other; })
      .def("__hash__",
           [](const qc::Operation& op) {
             return std::hash<qc::Operation>{}(op);
           })
      .def("__repr__", [](const qc::Operation& op) {
        std::ostringstream oss;
        oss << "Operation(type=" << op.getType() << ", ...)";
        return oss.str();
      });
}
} // namespace mqt
