#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "operations/CompoundOperation.hpp"
#include "operations/Control.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"
#include "operations/StandardOperation.hpp"
#include "operations/SymbolicOperation.hpp"
#include <iostream>
#include <sstream>
#include <memory>
namespace mqt {
namespace py = pybind11;
using namespace py::literals;
  void registerOperations(py::module& m) {

      py::class_<qc::Control>(m, "Control")
      .def(py::init<qc::Qubit>(), "qubit"_a, "Create a positive control qubit.")
      .def(py::init<qc::Qubit, qc::Control::Type>(), "qubit"_a, "type"_a,
           "Create a control qubit of the specified control type.")
      .def_readwrite(
          "control_type", &qc::Control::type,
          "The type of the control qubit. Can be positive or negative.")
      .def_readwrite("qubit", &qc::Control::qubit,
                     "The qubit index of the control qubit.");

  py::enum_<qc::Control::Type>(m, "ControlType")
      .value("Pos", qc::Control::Type::Pos)
      .value("Neg", qc::Control::Type::Neg)
      .export_values();
  py::implicitly_convertible<py::str, qc::Control::Type>();
  
      py::class_<qc::Operation>(m, "Operation", "Generic quantum operation.")
      .def_property(
          "targets", [](const qc::Operation& qc) { return qc.getTargets(); },
          [](qc::Operation& qc, const qc::Targets& tar) {
            return qc.setTargets(tar);
          })
      .def_property_readonly("n_targets", &qc::Operation::getNtargets)
      .def_property(
          "controls", [](const qc::Operation& qc) { return qc.getControls(); },
          [](qc::Operation& qc, const qc::Controls& tar) {
            return qc.setControls(tar);
          })
      .def_property_readonly("n_controls", &qc::Operation::getNcontrols)
      .def_property("n_qubits", &qc::Operation::getNqubits,
                    &qc::Operation::setNqubits)
      .def_property("name", &qc::Operation::getName, &qc::Operation::setName)
      .def("get_starting_qubit", &qc::Operation::getStartingQubit,
           "Get the starting qubit index of the operation.")
      .def("get_used_qubits", &qc::Operation::getUsedQubits,
           "Get the qubits used by the operation (both control and targets).")
      .def_property("gate", &qc::Operation::getType, &qc::Operation::setGate)
      .def("is_unitary", &qc::Operation::isUnitary)
      .def("is_standard_operation", &qc::Operation::isStandardOperation)
      .def("is_compound_operation", &qc::Operation::isCompoundOperation)
      .def("is_non_unitary_operation", &qc::Operation::isNonUnitaryOperation)
      .def("is_classic_controlled_operation",
           &qc::Operation::isClassicControlledOperation)
      .def("is_symbolic_operation", &qc::Operation::isSymbolicOperation)
      .def("is_controlled", &qc::Operation::isControlled)
      .def("acts_on", &qc::Operation::actsOn, "qubit"_a,
           "Check if the operation acts on the specified qubit.");

  py::class_<qc::StandardOperation, qc::Operation>(
      m, "StandardOperation",
      "Standard quantum operation."
      "This class is used to represent all standard operations, i.e. "
      "operations that can be represented by a single gate."
      "This includes all single qubit gates, as well as multi-qubit gates like "
      "CNOT, SWAP, etc. as well primitives like barriers and measurements.")
      .def(py::init<>(), "Create an empty standard operation. This is "
                         "equivalent to the identity gate.")
      .def(py::init<std::size_t, qc::Qubit, qc::OpType, std::vector<qc::fp>,
                    qc::Qubit>(),
           "nq"_a, "target"_a, "op_type"_a, "params"_a = std::vector<qc::fp>{},
           "starting_qubit"_a = 0,
           "Create a single-qubit standard operation of specified type.")
      .def(py::init<std::size_t, const qc::Targets&, qc::OpType,
                    std::vector<qc::fp>, qc::Qubit>(),
           "nq"_a, "targets"_a, "op_type"_a, "params"_a = std::vector<qc::fp>{},
           "starting_qubit"_a = 0,
           "Create a multi-qubit standard operation of specified type.")
      .def(py::init<std::size_t, qc::Control, qc::Qubit, qc::OpType,
                    const std::vector<qc::fp>&, qc::Qubit>(),
           "nq"_a, "control"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0,
           "Create a controlled standard operation of specified type.")
      .def(py::init<std::size_t, qc::Control, const qc::Targets&, qc::OpType,
                    const std::vector<qc::fp>&, qc::Qubit>(),
           "nq"_a, "control"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0,
           "Create a controlled multi-target standard operation of specified "
           "type.")
      .def(py::init<std::size_t, const qc::Controls&, qc::Qubit, qc::OpType,
                    const std::vector<qc::fp>&, qc::Qubit>(),
           "nq"_a, "controls"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0,
           "Create a multi-controlled standard operation of specified type.")
      .def(py::init<std::size_t, const qc::Controls&, const qc::Targets&,
                    qc::OpType, std::vector<qc::fp>, qc::Qubit>(),
           "nq"_a, "controls"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0,
           "Create a multi-controlled multi-target standard operation of "
           "specified type.")
      .def(py::init<std::size_t, const qc::Controls&, qc::Qubit, qc::Qubit>(),
           "nq"_a, "controls"_a, "target"_a, "starting_qubit"_a = 0,
           "Create a multi-controlled single-target operation of specified "
           "type involving nq consecutive control qubits starting_qubit.")
      .def(py::init<std::size_t, const qc::Controls&, qc::Qubit, qc::Qubit,
                    qc::OpType, std::vector<qc::fp>, qc::Qubit>(),
           "nq"_a, "controls"_a, "target0"_a, "target1"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0,
           "Create a multi-controlled two-target operation of specified type "
           "involving nq consecutive control qubits starting_qubit.")
      .def("is_standard_operation", &qc::StandardOperation::isStandardOperation)
      .def("clone", &qc::StandardOperation::clone,
           "Return deep clone of the operation.")
      .def("equals", py::overload_cast<const qc::Operation&>(
                         &qc::StandardOperation::equals, py::const_))
      .def("equals",
           py::overload_cast<const qc::Operation&, const qc::Permutation&,
                             const qc::Permutation&>(
               &qc::StandardOperation::equals, py::const_))
      .def("to_open_qasm",
           [](const qc::StandardOperation& op, const qc::RegisterNames& qreg,
              const qc::RegisterNames& creg) {
             std::ostringstream oss;
             op.dumpOpenQASM(oss, qreg, creg);
             return oss.str();
           });

  py::class_<qc::CompoundOperation, qc::Operation>(
      m, "CompoundOperation",
      "Quantum operation comprised of multiple sub-operations.")
    .def(py::init<std::size_t>(), "nq"_a, "Create an empty compound operation on `nq` qubits.")
      .def(py::init([](std::size_t nq, std::vector<qc::Operation*> ops) {
             std::vector<std::unique_ptr<qc::Operation>> unique_ops;
             unique_ops.reserve(ops.size());
             for (auto& op : ops) {
               unique_ops.emplace_back(std::move(op));
             }
             return qc::CompoundOperation(nq, std::move(unique_ops));
           }),
           "nq"_a, "ops"_a,
           "Create a compound operation from a list of operations.")
      .def("clone", &qc::CompoundOperation::clone,
           "Return deep clone of the operation.")
      .def("set_n_qubits", &qc::CompoundOperation::setNqubits)
      .def("is_compound_operation", &qc::CompoundOperation::isCompoundOperation)
      .def("is_non_unitary_operation",
           &qc::CompoundOperation::isNonUnitaryOperation)
      .def("equals",
           py::overload_cast<const qc::Operation&, const qc::Permutation&,
                             const qc::Permutation&>(
               &qc::CompoundOperation::equals, py::const_))
      .def("acts_on", &qc::CompoundOperation::actsOn)
      // .def("add_depth_contribution",
      //      &qc::CompoundOperation::addDepthContribution, "depths"_a,)
      .def("__len__", &qc::CompoundOperation::size,
           "Return number of sub-operations.")
      .def("empty", &qc::CompoundOperation::empty)
      .def("__getitem__", [](const qc::CompoundOperation& op,
                             std::size_t i) { return op.at(i).get(); }, py::return_value_policy::reference_internal, "i"_a, "Return i-th sub-operation. Beware: this gives write access to the sub-operation.")
      .def("get_used_qubits", &qc::CompoundOperation::getUsedQubits,
           "Return set of qubits used by the operation.")
      .def("to_open_qasm",
           [](const qc::CompoundOperation& op, const qc::RegisterNames& qreg,
              const qc::RegisterNames& creg) {
             std::ostringstream oss;
             op.dumpOpenQASM(oss, qreg, creg);
             return oss.str();
           })
      .def(
          "append_operation",
          [](qc::CompoundOperation& compOp, const qc::Operation& op) {
            compOp.emplace_back(op.clone()); 
          },
          "op"_a, "Append operation op to the `CompoundOperation`.");

  py::class_<qc::NonUnitaryOperation, qc::Operation>(
      m, "NonUnitaryOperation",
      "Non-unitary operations such as classically controlled quantum gates.")
      .def(
          py::init<std::size_t, std::vector<qc::Qubit>, std::vector<qc::Bit>>(),
          "nq"_a, "targets"_a, "classics"_a,
          "Create an nq qubit multi-qubit non-unitary operation controlled by "
          "a classical bit.")
      .def(py::init<std::size_t, qc::Qubit, qc::Bit>(), "nq"_a, "target"_a,
           "classic"_a,
           "Create an nq qubit non-unitary operation on qubit target "
           "controlled by a classical bit.")
      .def(py::init<std::size_t, std::vector<qc::Qubit>, qc::OpType>(), "nq"_a,
           "targets"_a, "op_type"_a,
           "Create an nq qubit multi-qubit non-unitary operation of specified "
           "type.")
      .def("clone", &qc::NonUnitaryOperation::clone,
           "Return deep clone of the operation.")
      .def("is_unitary", &qc::NonUnitaryOperation::isUnitary)
      .def("is_non_unitary_operation",
           &qc::NonUnitaryOperation::isNonUnitaryOperation)
      .def_property(
          "targets",
          py::overload_cast<>(&qc::NonUnitaryOperation::getTargets, py::const_),
          &qc::NonUnitaryOperation::setTargets, "Return the target qubits.")
      .def_property_readonly("n_targets", &qc::NonUnitaryOperation::getNtargets)
      .def_property_readonly(
          "classics",
          py::overload_cast<>(&qc::NonUnitaryOperation::getClassics,
                              py::const_),
          "Return the classical bits.")
      // .def("add_depth_contribution",
      //      &qc::NonUnitaryOperation::addDepthContribution)
      .def("acts_on", &qc::NonUnitaryOperation::actsOn,
           "Return set of qubits acted on by the operation.")
      .def("equals",
           py::overload_cast<const qc::Operation&, const qc::Permutation&,
                             const qc::Permutation&>(
               &qc::NonUnitaryOperation::equals, py::const_))
      .def("equals", py::overload_cast<const qc::Operation&>(
                         &qc::NonUnitaryOperation::equals, py::const_))
      .def("get_used_qubits", &qc::NonUnitaryOperation::getUsedQubits,
           "Return set of qubits used by the operation.")
      .def("to_open_qasm",
           [](const qc::NonUnitaryOperation& op, const qc::RegisterNames& qreg,
              const qc::RegisterNames& creg) {
             std::ostringstream oss;
             op.dumpOpenQASM(oss, qreg, creg);
             return oss.str();
           });

  py::class_<qc::Permutation>(m, "Permutation",
                              "Class representing a permutation of qubits.")
      .def("apply",
           py::overload_cast<const qc::Controls&>(&qc::Permutation::apply,
                                                  py::const_),
           "Apply the permutation to a set of controls and return the permuted "
           "controls.")
      .def("apply",
           py::overload_cast<const qc::Targets&>(&qc::Permutation::apply,
                                                 py::const_),
           "Apply the permutation to a set of targets and return the permuted "
           "targets.")
      .def("__getitem__",
           [](const qc::Permutation& p, qc::Qubit q) { return p.at(q); })
      .def("__setitem__",
           [](qc::Permutation& p, qc::Qubit q, qc::Qubit r) { p.at(q) = r; })
      .def(
          "__iter__",
          [](const qc::Permutation& p) {
            return py::make_iterator(p.begin(), p.end());
          },
          py::keep_alive<0, 1>());

  py::class_<qc::SymbolicOperation, qc::Operation>(
      m, "SymbolicOperation",
      "Class representing a symbolic operation."
      "This encompasses all symbolic versions of `StandardOperation` that "
      "involve (float) angle parameters.")
      .def(py::init<>(), "Create an empty symbolic operation.")
      .def(py::init<std::size_t, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&, qc::Qubit>(),
           "nq"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0,
           "Create a symbolic operation acting on a single qubit."
           "Params is a list of parameters that can be either `Expression` or "
           "`float`.")
      .def(py::init<std::size_t, const qc::Targets&, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&, qc::Qubit>(),
           "nq"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0,
           "Create a symbolic operation acting on multiple qubits."
           "Params is a list of parameters that can be either `Expression` or "
           "`float`.")
      .def(py::init<std::size_t, qc::Control, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&, qc::Qubit>(),
           "nq"_a, "control"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0,
           "Create a controlled symbolic operation."
           "Params is a list of parameters that can be either `Expression` or "
           "`float`.")
      .def(py::init<std::size_t, qc::Control, const qc::Targets&, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&, qc::Qubit>(),
           "nq"_a, "control"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0,
           "Create a controlled multi-target symbolic operation."
           "Params is a list of parameters that can be either `Expression` or "
           "`float`.")
      .def(py::init<std::size_t, const qc::Controls&, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&, qc::Qubit>(),
           "nq"_a, "controls"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0,
           "Create a multi-controlled symbolic operation."
           "Params is a list of parameters that can be either `Expression` or "
           "`float`.")
      .def(py::init<std::size_t, const qc::Controls&, const qc::Targets&,
                    qc::OpType, const std::vector<qc::SymbolOrNumber>&,
                    qc::Qubit>(),
           "nq"_a, "controls"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0,
           "Create a multi-controlled multi-target symbolic operation."
           "Params is a list of parameters that can be either `Expression` or "
           "`float`.")
      .def(py::init<std::size_t, const qc::Controls&, qc::Qubit, qc::Qubit,
                    qc::OpType, const std::vector<qc::SymbolOrNumber>&,
                    qc::Qubit>(),
           "nq"_a, "controls"_a, "target0"_a, "target1"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0,
           "Create a multi-controlled two-target symbolic operation."
           "Params is a list of parameters that can be either `Expression` or "
           "`float`.")
      .def("get_parameter", &qc::SymbolicOperation::getParameter)
      .def("get_parameters", &qc::SymbolicOperation::getParameters)
      .def("clone", &qc::SymbolicOperation::clone,
           "Create a deep copy of this operation.")
      .def("is_symbolic_operation", &qc::SymbolicOperation::isSymbolicOperation,
           "Return true if this operation is actually parameterized by a "
           "symbolic parameter.")
      .def("is_standard_operation", &qc::SymbolicOperation::isStandardOperation,
           "Return true if this operation is not parameterized by a symbolic "
           "parameter.")
      .def("equals",
           py::overload_cast<const qc::Operation&, const qc::Permutation&,
                             const qc::Permutation&>(
               &qc::SymbolicOperation::equals, py::const_))
      .def("equals", py::overload_cast<const qc::Operation&>(
                         &qc::SymbolicOperation::equals, py::const_))
      .def("get_instantiated_operation",
           &qc::SymbolicOperation::getInstantiatedOperation,
           "assignment"_a
           "Return a `StandardOperation` version of this operation that is "
           "obtainedby replacing all variables by their values dictated by the "
           "dict assignment which maps Variable objects to float.")
      .def("instantiate", &qc::SymbolicOperation::instantiate,
           "assignment"_a
           "Replace all variables within this operation by their values "
           "dictated by the dict assignment which maps Variable objects to "
           "float.");

  }
}
