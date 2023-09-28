#include "Definitions.hpp"
#include "Permutation.hpp"
#include "QuantumComputation.hpp"
#include "operations/CompoundOperation.hpp"
#include "operations/Control.hpp"
#include "operations/Expression.hpp"
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
namespace py = pybind11;
using namespace pybind11::literals;

namespace mqt {

PYBIND11_MODULE(_core, m) {

  py::class_<qc::QuantumComputation>(
      m, "QuantumComputation",
      "Representation of quantum circuits within MQT Core")
      .def(py::init<std::size_t>(), "nq"_a,
           "Constructs an empty QuantumComputation with the given number of "
           "qubits.")
      .def(py::init<std::string>(), "filename"_a,
           "Read QuantumComputation from given file. Supported formats are "
           "[OpenQASM, Real, GRCS, TFC, QC]")
      // .def_rw("name", &qc::QuantumComputation::getName,
      // &qc::QuantumComputation::setName)
      .def("clone", &qc::QuantumComputation::clone,
           "Clone this QuantumComputation object.")
      .def_property_readonly("n_qubits", &qc::QuantumComputation::getNqubits)
      .def_property_readonly("n_ancillae",
                             &qc::QuantumComputation::getNancillae)
      .def_property_readonly("n_qubits_without_ancillae",
                             &qc::QuantumComputation::getNqubitsWithoutAncillae)
      .def_property_readonly("n_cbits", &qc::QuantumComputation::getNcbits)
      .def_property_readonly("n_ops", &qc::QuantumComputation::getNops)
      .def_property_readonly("n_single_qubit_ops",
                             &qc::QuantumComputation::getNsingleQubitOps)
      .def_property_readonly("n_individual_ops",
                             &qc::QuantumComputation::getNindividualOps)
      .def_property_readonly("depth", &qc::QuantumComputation::getDepth)
      .def_property("gphase", &qc::QuantumComputation::getGlobalPhase,
                    &qc::QuantumComputation::gphase)
      .def("i", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::i))
      .def("i", py::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::i))
      .def("i", py::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::i))
      .def("h", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::h))
      .def("h", py::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::h))
      .def("h", py::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::h))
      .def("x", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::x))
      .def("x", py::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::x))
      .def("x", py::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::x))
      .def("y", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::y))
      .def("y", py::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::y))
      .def("y", py::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::y))
      .def("z", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::z))
      .def("z", py::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::z))
      .def("z", py::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::z))
      .def("s", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::s))
      .def("s", py::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::s))
      .def("s", py::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::s))
      .def("sdag", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::sdag))
      .def("sdag", py::overload_cast<qc::Qubit, const qc::Control&>(
                       &qc::QuantumComputation::sdag))
      .def("sdag", py::overload_cast<qc::Qubit, const qc::Controls&>(
                       &qc::QuantumComputation::sdag))
      .def("t", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::t))
      .def("t", py::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::t))
      .def("t", py::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::t))
      .def("tdag", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::tdag))
      .def("tdag", py::overload_cast<qc::Qubit, const qc::Control&>(
                       &qc::QuantumComputation::tdag))
      .def("tdag", py::overload_cast<qc::Qubit, const qc::Controls&>(
                       &qc::QuantumComputation::tdag))
      .def("v", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::v))
      .def("v", py::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::v))
      .def("v", py::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::v))
      .def("vdag", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::vdag))
      .def("vdag", py::overload_cast<qc::Qubit, const qc::Control&>(
                       &qc::QuantumComputation::vdag))
      .def("vdag", py::overload_cast<qc::Qubit, const qc::Controls&>(
                       &qc::QuantumComputation::vdag))
      .def("u3", py::overload_cast<qc::Qubit, const qc::fp, const qc::fp,
                                   const qc::fp>(&qc::QuantumComputation::u3))
      .def("u3", py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp,
                                   const qc::fp, const qc::fp>(
                     &qc::QuantumComputation::u3))
      .def("u3", py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp,
                                   const qc::fp, const qc::fp>(
                     &qc::QuantumComputation::u3))
      .def("u2", py::overload_cast<qc::Qubit, const qc::fp, const qc::fp>(
                     &qc::QuantumComputation::u2))
      .def("u2", py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp,
                                   const qc::fp>(&qc::QuantumComputation::u2))
      .def("u2", py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp,
                                   const qc::fp>(&qc::QuantumComputation::u2))
      .def("phase", py::overload_cast<qc::Qubit, const qc::fp>(
                        &qc::QuantumComputation::phase))
      .def("phase",
           py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
               &qc::QuantumComputation::phase))
      .def("phase",
           py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
               &qc::QuantumComputation::phase))
      .def("sx", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::sx))
      .def("sx", py::overload_cast<qc::Qubit, const qc::Control&>(
                     &qc::QuantumComputation::sx))
      .def("sx", py::overload_cast<qc::Qubit, const qc::Controls&>(
                     &qc::QuantumComputation::sx))
      .def("sxdag",
           py::overload_cast<qc::Qubit>(&qc::QuantumComputation::sxdag))
      .def("sxdag", py::overload_cast<qc::Qubit, const qc::Control&>(
                        &qc::QuantumComputation::sxdag))
      .def("sxdag", py::overload_cast<qc::Qubit, const qc::Controls&>(
                        &qc::QuantumComputation::sxdag))
      .def("rx", py::overload_cast<qc::Qubit, const qc::fp>(
                     &qc::QuantumComputation::rx))
      .def("rx", py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
                     &qc::QuantumComputation::rx))
      .def("rx",
           py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
               &qc::QuantumComputation::rx))
      .def("ry", py::overload_cast<qc::Qubit, const qc::fp>(
                     &qc::QuantumComputation::ry))
      .def("ry", py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
                     &qc::QuantumComputation::ry))
      .def("ry",
           py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
               &qc::QuantumComputation::ry))
      .def("rz", py::overload_cast<qc::Qubit, const qc::fp>(
                     &qc::QuantumComputation::rz))
      .def("rz", py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
                     &qc::QuantumComputation::rz))
      .def("rz",
           py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
               &qc::QuantumComputation::rz))
      .def("swap", py::overload_cast<qc::Qubit, qc::Qubit>(
                       &qc::QuantumComputation::swap))
      .def("swap", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                       &qc::QuantumComputation::swap))
      .def("swap", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                       &qc::QuantumComputation::swap))
      .def("iswap", py::overload_cast<qc::Qubit, qc::Qubit>(
                        &qc::QuantumComputation::iswap))
      .def("iswap", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                        &qc::QuantumComputation::iswap))
      .def("iswap",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
               &qc::QuantumComputation::iswap))
      .def("peres", py::overload_cast<qc::Qubit, qc::Qubit>(
                        &qc::QuantumComputation::peres))
      .def("peres", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                        &qc::QuantumComputation::peres))
      .def("peres",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
               &qc::QuantumComputation::peres))
      .def("peresdag", py::overload_cast<qc::Qubit, qc::Qubit>(
                           &qc::QuantumComputation::peresdag))
      .def("peresdag",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
               &qc::QuantumComputation::peresdag))
      .def("peresdag",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
               &qc::QuantumComputation::peresdag))
      .def("dcx", py::overload_cast<qc::Qubit, qc::Qubit>(
                      &qc::QuantumComputation::dcx))
      .def("dcx", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                      &qc::QuantumComputation::dcx))
      .def("dcx", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                      &qc::QuantumComputation::dcx))
      .def("ecr", py::overload_cast<qc::Qubit, qc::Qubit>(
                      &qc::QuantumComputation::ecr))
      .def("ecr", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                      &qc::QuantumComputation::ecr))
      .def("ecr", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                      &qc::QuantumComputation::ecr))
      .def("rxx", py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                      &qc::QuantumComputation::rxx))
      .def("rxx",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
               &qc::QuantumComputation::rxx))
      .def("rxx",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
               &qc::QuantumComputation::rxx))
      .def("ryy", py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                      &qc::QuantumComputation::ryy))
      .def("ryy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
               &qc::QuantumComputation::ryy))
      .def("ryy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
               &qc::QuantumComputation::ryy))
      .def("rzz", py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                      &qc::QuantumComputation::rzz))
      .def("rzz",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
               &qc::QuantumComputation::rzz))
      .def("rzz",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
               &qc::QuantumComputation::rzz))
      .def("rzx", py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                      &qc::QuantumComputation::rzx))
      .def("rzx",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
               &qc::QuantumComputation::rzx))
      .def("rzx",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
               &qc::QuantumComputation::rzx))
      .def("xx_minus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, qc::fp, qc::fp>(
               &qc::QuantumComputation::xx_minus_yy))
      .def("xx_minus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp,
                             qc::fp>(&qc::QuantumComputation::xx_minus_yy))
      .def("xx_minus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp,
                             qc::fp>(&qc::QuantumComputation::xx_minus_yy))
      .def("xx_plus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, qc::fp, qc::fp>(
               &qc::QuantumComputation::xx_plus_yy))
      .def("xx_plus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp,
                             qc::fp>(&qc::QuantumComputation::xx_plus_yy))
      .def("xx_plus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp,
                             qc::fp>(&qc::QuantumComputation::xx_plus_yy))
      .def("measure", py::overload_cast<qc::Qubit, std::size_t>(
                          &qc::QuantumComputation::measure))
      .def("measure",
           py::overload_cast<qc::Qubit, const std::pair<std::string, qc::Bit>&>(
               &qc::QuantumComputation::measure))
      .def("measure", py::overload_cast<const std::vector<qc::Qubit>&,
                                        const std::vector<qc::Bit>&>(
                          &qc::QuantumComputation::measure))
      .def("reset",
           py::overload_cast<qc::Qubit>(&qc::QuantumComputation::reset))
      .def("reset", py::overload_cast<const std::vector<qc::Qubit>&>(
                        &qc::QuantumComputation::reset))
      .def("barrier",
           py::overload_cast<qc::Qubit>(&qc::QuantumComputation::barrier))
      .def("barrier", py::overload_cast<const std::vector<qc::Qubit>&>(
                          &qc::QuantumComputation::barrier))
      .def("reset",
           py::overload_cast<qc::Qubit>(&qc::QuantumComputation::reset))
      .def("classic_controlled",
           py::overload_cast<const qc::OpType, const qc::Qubit,
                             const qc::ClassicalRegister&, const std::uint64_t,
                             const std::vector<qc::fp>&>(
               &qc::QuantumComputation::classicControlled))
      .def("classic_controlled",
           py::overload_cast<const qc::OpType, const qc::Qubit,
                             const qc::Control, const qc::ClassicalRegister&,
                             const std::uint64_t, const std::vector<qc::fp>&>(
               &qc::QuantumComputation::classicControlled))
      .def("classic_controlled",
           py::overload_cast<const qc::OpType, const qc::Qubit,
                             const qc::Controls&, const qc::ClassicalRegister&,
                             const std::uint64_t, const std::vector<qc::fp>&>(
               &qc::QuantumComputation::classicControlled))
      .def("set_logical_qubit_ancillary",
           &qc::QuantumComputation::setLogicalQubitAncillary)
      .def("add_qubit_register", &qc::QuantumComputation::addQubitRegister)
      .def("add_classical_bit_register",
           &qc::QuantumComputation::addClassicalRegister)
      .def("add_ancillary_register",
           &qc::QuantumComputation::addAncillaryRegister)
      .def("append_operation",
           [](qc::QuantumComputation& qc, const qc::Operation& op) {
             qc.emplace_back(op.clone()); // not an ideal solution but it works
           }) // Transfers ownership from Python to C++
      .def("instantiate", &qc::QuantumComputation::instantiate)
      .def("add_variable", &qc::QuantumComputation::addVariable)
      .def("add_variables",
           [](qc::QuantumComputation& qc,
              const std::vector<qc::SymbolOrNumber>& vars) {
             for (const auto& var : vars) {
               qc.addVariable(var);
             }
           })
      .def("is_variable_free", &qc::QuantumComputation::isVariableFree)
      .def("get_variables", &qc::QuantumComputation::getVariables)
      .def("initialize_io_mapping",
           &qc::QuantumComputation::initializeIOMapping)
      .def("from_file", py::overload_cast<const std::string&>(
                            &qc::QuantumComputation::import))
      .def("from_file",
           [](qc::QuantumComputation& qc, const std::string& filename,
              const std::string& format) {
             if (format == "qasm") {
               qc.import(filename, qc::Format::OpenQASM);
             } else if (format == "real") {
               qc.import(filename, qc::Format::Real);
             } else if (format == "grcs") {
               qc.import(filename, qc::Format::GRCS);
             } else if (format == "tfc") {
               qc.import(filename, qc::Format::TFC);
             } else if (format == "tensor") {
               qc.import(filename, qc::Format::Tensor);
             } else if (format == "qc") {
               qc.import(filename, qc::Format::QC);
             } else {
               throw qc::QFRException("Unknown format: " + format);
             }
           })
      .def("dump",
           py::overload_cast<const std::string&>(&qc::QuantumComputation::dump))
      .def("dump",
           [](qc::QuantumComputation& qc, const std::string& filename,
              const std::string& format) {
             if (format == "qasm") {
               qc.dump(filename, qc::Format::OpenQASM);
             } else if (format == "real") {
               qc.dump(filename, qc::Format::Real);
             } else if (format == "grcs") {
               qc.dump(filename, qc::Format::GRCS);
             } else if (format == "tfc") {
               qc.dump(filename, qc::Format::TFC);
             } else if (format == "tensor") {
               qc.dump(filename, qc::Format::Tensor);
             } else if (format == "qc") {
               qc.dump(filename, qc::Format::QC);
             } else {
               throw qc::QFRException("Unknown format: " + format);
             }
           })
      .def("to_open_qasm",
           [](qc::QuantumComputation& qc) {
             std::ostringstream oss;
             qc.dumpOpenQASM(oss);
             return oss.str();
           })
      .def("__len__", &qc::QuantumComputation::getNindividualOps)
      .def("__getitem__", [](const qc::QuantumComputation& qc,
                             std::size_t idx) { return qc.at(idx).get(); });
  py::class_<qc::Control>(m, "Control")
      .def(py::init<qc::Qubit>())
      .def(py::init<qc::Qubit, qc::Control::Type>())
      .def_readwrite("control_type", &qc::Control::type)
      .def_readwrite("qubit", &qc::Control::qubit);

  py::enum_<qc::Control::Type>(m, "ControlType")

      .value("Pos", qc::Control::Type::Pos)
      .value("Neg", qc::Control::Type::Neg)
      .export_values();

  py::class_<qc::Operation>(m, "Operation")
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
      .def("get_starting_qubit", &qc::Operation::getStartingQubit)
      .def("get_used_qubits", &qc::Operation::getUsedQubits)
      .def_property("gate", &qc::Operation::getType, &qc::Operation::setGate)
      .def("is_unitary", &qc::Operation::isUnitary)
      .def("is_standard_operation", &qc::Operation::isStandardOperation)
      .def("is_compound_operation", &qc::Operation::isCompoundOperation)
      .def("is_non_unitary_operation", &qc::Operation::isNonUnitaryOperation)
      .def("is_classic_controlled_operation",
           &qc::Operation::isClassicControlledOperation)
      .def("is_symbolic_operation", &qc::Operation::isSymbolicOperation)
      .def("is_controlled", &qc::Operation::isControlled)
      .def("acts_on", &qc::Operation::actsOn);
  ;
  py::class_<qc::StandardOperation, qc::Operation>(m, "StandardOperation")
      .def(py::init<>())
      .def(py::init<std::size_t, qc::Qubit, qc::OpType, std::vector<qc::fp>,
                    qc::Qubit>(),
           "nq"_a, "target"_a, "op_type"_a, "params"_a = std::vector<qc::fp>{},
           "starting_qubit"_a = 0)
      .def(py::init<std::size_t, const qc::Targets&, qc::OpType,
                    std::vector<qc::fp>, qc::Qubit>(),
           "nq"_a, "targets"_a, "op_type"_a, "params"_a = std::vector<qc::fp>{},
           "starting_qubit"_a = 0)
      .def(py::init<std::size_t, qc::Control, qc::Qubit, qc::OpType,
                    const std::vector<qc::fp>&, qc::Qubit>(),
           "nq"_a, "control"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0)
      .def(py::init<std::size_t, qc::Control, const qc::Targets&, qc::OpType,
                    const std::vector<qc::fp>&, qc::Qubit>(),
           "nq"_a, "control"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0)
      .def(py::init<std::size_t, const qc::Controls&, qc::Qubit, qc::OpType,
                    const std::vector<qc::fp>&, qc::Qubit>(),
           "nq"_a, "controls"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0)
      .def(py::init<std::size_t, const qc::Controls&, const qc::Targets&,
                    qc::OpType, std::vector<qc::fp>, qc::Qubit>(),
           "nq"_a, "controls"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0)
      .def(py::init<std::size_t, const qc::Controls&, qc::Qubit, qc::Qubit>(),
           "nq"_a, "controls"_a, "target"_a, "starting_qubit"_a = 0)
      .def(py::init<std::size_t, const qc::Controls&, qc::Qubit, qc::Qubit,
                    qc::OpType, std::vector<qc::fp>, qc::Qubit>(),
           "nq"_a, "controls"_a, "target0"_a, "target1"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0)
      .def("is_standard_operation", &qc::StandardOperation::isStandardOperation)
      .def("clone", &qc::StandardOperation::clone)
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

  py::class_<qc::CompoundOperation, qc::Operation>(m, "CompoundOperation")
      .def(py::init([](std::size_t nq, std::vector<qc::Operation*> ops) {
        std::vector<std::unique_ptr<qc::Operation>> unique_ops;
        unique_ops.reserve(ops.size());
        for (auto& op : ops) {
          unique_ops.emplace_back(std::move(op));
        }
        return qc::CompoundOperation(nq, std::move(unique_ops));
      }))
      .def("clone", &qc::CompoundOperation::clone)
      .def("set_n_qubits", &qc::CompoundOperation::setNqubits)
      .def("is_compound_operation", &qc::CompoundOperation::isCompoundOperation)
      .def("is_non_unitary_operation",
           &qc::CompoundOperation::isNonUnitaryOperation)
      .def("equals",
           py::overload_cast<const qc::Operation&, const qc::Permutation&,
                             const qc::Permutation&>(
               &qc::CompoundOperation::equals, py::const_))
      .def("acts_on", &qc::CompoundOperation::actsOn)
      .def("add_depth_contribution",
           &qc::CompoundOperation::addDepthContribution)
      .def("__len__", &qc::CompoundOperation::size)
      .def("size", &qc::CompoundOperation::size)
      .def("empty", &qc::CompoundOperation::empty)
      .def("__getitem__", [](const qc::CompoundOperation& op,
                             std::size_t i) { return op.at(i).get(); })
      .def("get_used_qubits", &qc::CompoundOperation::getUsedQubits)
      .def("to_open_qasm",
           [](const qc::CompoundOperation& op, const qc::RegisterNames& qreg,
              const qc::RegisterNames& creg) {
             std::ostringstream oss;
             op.dumpOpenQASM(oss, qreg, creg);
             return oss.str();
           });

  py::class_<qc::NonUnitaryOperation, qc::Operation>(m, "NonUnitaryOperation")
      .def(
          py::init<std::size_t, std::vector<qc::Qubit>, std::vector<qc::Bit>>())
      .def(py::init<std::size_t, qc::Qubit, qc::Bit>())
      .def(py::init<std::size_t, std::vector<qc::Qubit>, qc::OpType>())
      .def("clone", &qc::NonUnitaryOperation::clone)
      .def("is_unitary", &qc::NonUnitaryOperation::isUnitary)
      .def("is_non_unitary_operation",
           &qc::NonUnitaryOperation::isNonUnitaryOperation)
      .def_property(
          "targets",
          py::overload_cast<>(&qc::NonUnitaryOperation::getTargets, py::const_),
          &qc::NonUnitaryOperation::setTargets)
      .def_property_readonly("n_targets", &qc::NonUnitaryOperation::getNtargets)
      .def_property_readonly(
          "classics", py::overload_cast<>(&qc::NonUnitaryOperation::getClassics,
                                          py::const_))
      .def("add_depth_contribution",
           &qc::NonUnitaryOperation::addDepthContribution)
      .def("acts_on", &qc::NonUnitaryOperation::actsOn)
      .def("equals",
           py::overload_cast<const qc::Operation&, const qc::Permutation&,
                             const qc::Permutation&>(
               &qc::NonUnitaryOperation::equals, py::const_))
      .def("equals", py::overload_cast<const qc::Operation&>(
                         &qc::NonUnitaryOperation::equals, py::const_))
      .def("get_used_qubits", &qc::NonUnitaryOperation::getUsedQubits)
      .def("to_open_qasm",
           [](const qc::NonUnitaryOperation& op, const qc::RegisterNames& qreg,
              const qc::RegisterNames& creg) {
             std::ostringstream oss;
             op.dumpOpenQASM(oss, qreg, creg);
             return oss.str();
           });

  py::class_<qc::Permutation>(m, "Permutation")
      .def("apply", py::overload_cast<const qc::Controls&>(
                        &qc::Permutation::apply, py::const_))
      .def("apply", py::overload_cast<const qc::Targets&>(
                        &qc::Permutation::apply, py::const_))
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

  py::class_<qc::SymbolicOperation, qc::Operation>(m, "SymbolicOperation")
      .def(py::init<>())
      .def(py::init<std::size_t, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&, qc::Qubit>(),
           "nq"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0)
      .def(py::init<std::size_t, const qc::Targets&, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&, qc::Qubit>(),
           "nq"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0)
      .def(py::init<std::size_t, qc::Control, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&, qc::Qubit>(),
           "nq"_a, "control"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0)
      .def(py::init<std::size_t, qc::Control, const qc::Targets&, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&, qc::Qubit>(),
           "nq"_a, "control"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0)
      .def(py::init<std::size_t, const qc::Controls&, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&, qc::Qubit>(),
           "nq"_a, "controls"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0)
      .def(py::init<std::size_t, const qc::Controls&, const qc::Targets&,
                    qc::OpType, const std::vector<qc::SymbolOrNumber>&,
                    qc::Qubit>(),
           "nq"_a, "controls"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0)
      .def(py::init<std::size_t, const qc::Controls&, qc::Qubit, qc::Qubit,
                    qc::OpType, const std::vector<qc::SymbolOrNumber>&,
                    qc::Qubit>(),
           "nq"_a, "controls"_a, "target0"_a, "target1"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{},
           "starting_qubit"_a = 0)
      .def("get_parameter", &qc::SymbolicOperation::getParameter)
      .def("get_parameters", &qc::SymbolicOperation::getParameters)
      .def("clone", &qc::SymbolicOperation::clone)
      .def("is_symbolic_operation", &qc::SymbolicOperation::isSymbolicOperation)
      .def("is_standard_operation", &qc::SymbolicOperation::isStandardOperation)
      .def("equals",
           py::overload_cast<const qc::Operation&, const qc::Permutation&,
                             const qc::Permutation&>(
               &qc::SymbolicOperation::equals, py::const_))
      .def("equals", py::overload_cast<const qc::Operation&>(
                         &qc::SymbolicOperation::equals, py::const_))
      .def("get_instantiated_operation",
           &qc::SymbolicOperation::getInstantiatedOperation)
      .def("instantiate", &qc::SymbolicOperation::instantiate);

  py::class_<sym::Variable>(m, "Variable")
      .def(py::init<std::string>())
      .def_property_readonly("name", &sym::Variable::getName)
      .def("__str__", &sym::Variable::getName)
      .def("__eq__", &sym::Variable::operator==)
      .def("__ne__", &sym::Variable::operator!=)
      .def("__lt__", &sym::Variable::operator<)
      .def("__gt__", &sym::Variable::operator>);

  py::class_<sym::Term<double>>(m, "Term")
      .def(py::init<double, sym::Variable>())
      .def(py::init<sym::Variable>())
      .def_property_readonly("variable", &sym::Term<double>::getVar)
      .def_property_readonly("coefficient", &sym::Term<double>::getCoeff)
      .def("has_zero_coefficient", &sym::Term<double>::hasZeroCoeff)
      .def("add_coefficient", &sym::Term<double>::addCoeff)
      .def("total_assignment", &sym::Term<double>::totalAssignment)
      .def("evaluate", &sym::Term<double>::evaluate)
      .def("__mul__",
           [](const sym::Term<double>& lhs, double rhs) { return lhs * rhs; })
      .def("__rmul__",
           [](sym::Term<double> rhs, const double lhs) { return lhs * rhs; })
      .def("__truediv__",
           [](const sym::Term<double>& lhs, double rhs) { return lhs / rhs; })
      .def("__rtruediv__",
           [](sym::Term<double> rhs, const double lhs) { return lhs / rhs; });

  py::class_<sym::Expression<double, double>>(m, "Expression")
      .def(py::init<>())
      .def("__init__",
           [](sym::Expression<double, double>* expr,
              const std::vector<sym::Term<double>>& terms, double constant) {
             new (expr) sym::Expression<double, double>(terms, constant);
           })
      .def("__init__",
           [](sym::Expression<double, double>* expr,
              const sym::Term<double>& term, double constant) {
             new (expr) sym::Expression<double, double>(
                 std::vector<sym::Term<double>>{term}, constant);
           })
      .def(py::init<double>())
      .def_property("constant", &sym::Expression<double, double>::getConst,
                    &sym::Expression<double, double>::setConst)
      .def(
          "__iter__",
          [](const sym::Expression<double, double>& expr) {
            return py::make_iterator(expr.begin(), expr.end());
          },
          py::keep_alive<0, 1>())
      .def("__getitem__",
           [](const sym::Expression<double, double>& expr, std::size_t idx) {
             if (idx >= expr.numTerms()) {
               throw py::index_error();
             }
             return expr.getTerms()[idx];
           })

      .def("is_zero", &sym::Expression<double, double>::isZero)
      .def("is_constant", &sym::Expression<double, double>::isConstant)
      .def("num_terms", &sym::Expression<double, double>::numTerms)
      .def("__len__", &sym::Expression<double, double>::numTerms)
      .def_property_readonly("terms",
                             &sym::Expression<double, double>::getTerms)
      .def("evaluate", &sym::Expression<double, double>::evaluate)
      // addition operators
      .def("__add__",
           [](const sym::Expression<double, double>& lhs,
              const sym::Expression<double, double>& rhs) { return lhs + rhs; })
      .def("__add__", [](const sym::Expression<double, double>& lhs,
                         const sym::Term<double>& rhs) { return lhs + rhs; })
      .def("__add__", [](const sym::Expression<double, double>& lhs,
                         const double rhs) { return lhs + rhs; })
      .def("__radd__", [](const sym::Expression<double, double>& rhs,
                          const sym::Term<double>& lhs) { return lhs + rhs; })
      .def("__radd__", [](const sym::Expression<double, double>& rhs,
                          const double lhs) { return rhs + lhs; })
      // subtraction operators
      .def("__sub__",
           [](const sym::Expression<double, double>& lhs,
              const sym::Expression<double, double>& rhs) { return lhs - rhs; })
      .def("__sub__", [](const sym::Expression<double, double>& lhs,
                         const sym::Term<double>& rhs) { return lhs - rhs; })
      .def("__sub__", [](const sym::Expression<double, double>& lhs,
                         const double rhs) { return lhs - rhs; })
      .def("__rsub__", [](const sym::Expression<double, double>& rhs,
                          const sym::Term<double>& lhs) { return lhs - rhs; })
      .def("__rsub__", [](const sym::Expression<double, double>& rhs,
                          const double lhs) { return lhs - rhs; })
      // multiplication operators
      .def("__mul__", [](const sym::Expression<double, double>& lhs,
                         double rhs) { return lhs * rhs; })
      .def("__rmul__", [](const sym::Expression<double, double>& rhs,
                          double lhs) { return rhs * lhs; })
      // division operators
      .def("__truediv__", [](const sym::Expression<double, double>& lhs,
                             double rhs) { return lhs / rhs; })
      .def("__rtruediv__", [](const sym::Expression<double, double>& rhs,
                              double lhs) { return rhs / lhs; })

      .def(
          "__eq__",
          [](const sym::Expression<double, double>& lhs,
             const sym::Expression<double, double>& rhs) { return lhs == rhs; })
      .def("__str__", [](const sym::Expression<double, double>& expr) {
        std::stringstream ss;
        ss << expr;
        return ss.str();
      });

  py::enum_<qc::OpType>(m, "OpType")
      .value("none", qc::OpType::None)
      .value("gphase", qc::OpType::GPhase)
      .value("i", qc::OpType::I)
      .value("h", qc::OpType::H)
      .value("x", qc::OpType::X)
      .value("y", qc::OpType::Y)
      .value("z", qc::OpType::Z)
      .value("s", qc::OpType::S)
      .value("sdag", qc::OpType::Sdag)
      .value("t", qc::OpType::T)
      .value("tdag", qc::OpType::Tdag)
      .value("v", qc::OpType::V)
      .value("vdag", qc::OpType::Vdag)
      .value("u3", qc::OpType::U3)
      .value("u2", qc::OpType::U2)
      .value("phase", qc::OpType::Phase)
      .value("sx", qc::OpType::SX)
      .value("sxdag", qc::OpType::SXdag)
      .value("rx", qc::OpType::RX)
      .value("ry", qc::OpType::RY)
      .value("rz", qc::OpType::RZ)
      .value("swap", qc::OpType::SWAP)
      .value("iswap", qc::OpType::iSWAP)
      .value("peres", qc::OpType::Peres)
      .value("peresdag", qc::OpType::Peresdag)
      .value("dcx", qc::OpType::DCX)
      .value("ecr", qc::OpType::ECR)
      .value("rxx", qc::OpType::RXX)
      .value("ryy", qc::OpType::RYY)
      .value("rzz", qc::OpType::RZZ)
      .value("rzx", qc::OpType::RZX)
      .value("xx_minus_yy", qc::OpType::XXminusYY)
      .value("xx_plus_yy", qc::OpType::XXplusYY)
      .value("compound", qc::OpType::Compound)
      .value("measure", qc::OpType::Measure)
      .value("reset", qc::OpType::Reset)
      .value("barrier", qc::OpType::Barrier)
      .value("teleportation", qc::OpType::Teleportation)
      .value("classiccontrolled", qc::OpType::ClassicControlled)
      .export_values()
      .def("__str__", [](const qc::OpType& op) { return qc::toString(op); })
      .def_static("from_string",
                  [](const std::string& s) { return qc::opTypeFromString(s); });
}

} // namespace mqt
