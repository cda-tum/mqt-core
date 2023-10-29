#include "operations/Operation.hpp"
#include "python/pybind11.hpp"

#include <pybind11/operators.h>

namespace mqt {

void registerOperation(py::module& m) {
  py::class_<qc::Operation>(m, "Operation", "Generic quantum operation.")
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
      .def("add_control", &qc::Operation::addControl, "control"_a,
           "Add a control to this operation.")
      .def("add_controls", &qc::Operation::addControls, "controls"_a,
           "Add a list of controls to this operation.")
      .def("clear_controls", &qc::Operation::clearControls,
           "Remove all controls from this operation.")
      .def(
          "remove_control",
          [](qc::Operation& op, const qc::Control& c) { op.removeControl(c); },
          "control"_a, "Remove a control from this operation.")
      .def("remove_controls", &qc::Operation::removeControls, "controls"_a,
           "Remove a list of controls from this operation.")
      .def_property("num_qubits", &qc::Operation::getNqubits,
                    &qc::Operation::setNqubits)
      .def("get_used_qubits", &qc::Operation::getUsedQubits,
           "Get the qubits used by the operation (both control and targets).")
      .def("acts_on", &qc::Operation::actsOn, "qubit"_a,
           "Check if the operation acts on the specified qubit.")
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
      .def("get_inverted", &qc::Operation::getInverted,
           "Return the inverse of this operation.")
      .def("invert", &qc::Operation::invert, "Invert this operation.")
      .def(
          "qasm_str",
          [](const qc::Operation& op, const qc::RegisterNames& qreg,
             const qc::RegisterNames& creg) {
            std::ostringstream oss;
            op.dumpOpenQASM(oss, qreg, creg);
            return oss.str();
          },
          "qreg"_a, "creg"_a,
          "Return the OpenQASM string representation of this operation.")
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(hash(py::self))
      .def("__str__",
           [](const qc::Operation& op) {
             std::ostringstream oss;
             oss << op;
             return oss.str();
           })
      .def("__repr__", [](const qc::Operation& op) {
        std::ostringstream oss;
        oss << op;
        return oss.str();
      });
}
} // namespace mqt
