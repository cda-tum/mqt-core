#include "Definitions.hpp"
#include "Permutation.hpp"
#include "QuantumComputation.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"
#include "operations/StandardOperation.hpp"

#include <cstddef>
#include <iostream>
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
      // .def("__str__", [](const qc::QuantumComputation& qc){std::stringstream
      // ss;qc.print(ss);return ss.str();})
      // .def("__iter__", [](const qc::QuantumComputation& qc) {
      //   return py::make_iterator(py::type<qc::QuantumComputation>(), "ops",
      //   qc.begin(), qc.end());
      // }, py::keep_alive<0, 1>())
      .def("__len__", &qc::QuantumComputation::getNindividualOps)
      .def("__getitem__", [](const qc::QuantumComputation& qc,
                             std::size_t idx) { return qc.at(idx).get(); })
      // .def_property_readonly("ops", [](const qc::QuantumComputation& qc,
      // std::size_t idx){return *qc.at(idx);})
      ;
  py::class_<qc::Control>(m, "Control")
      .def_readwrite("type", &qc::Control::type)
      .def_readwrite("qubit", &qc::Control::qubit)
      .def(py::init<qc::Qubit>())
      .def(py::init<qc::Qubit, qc::Control::Type>());

  py::enum_<qc::Control::Type>(m, "ControlType")
      .value("Pos", qc::Control::Type::Pos)
      .value("Neg", qc::Control::Type::Neg)
      .export_values();

  py::class_<qc::Operation>(m, "Operation")
      // .def(py::init<>())
      // .def_property("targets", &qc::Operation::getTargets,
      // &qc::Operation::setTargets)
      .def_property("name", &qc::Operation::getName, &qc::Operation::setName)
      .def("set_gate", &qc::Operation::setGate);
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
      // .def("__str__", [](const qc::StandardOperation& qc) {
      //   std::ostringstream ss;
      //   qc.dumpOpenQASM(ss, {{"q", "0"}, {"q", "1"}, {"q", "2"}}, {});
      //   ss.str();
      // })
      ;

  py::class_<qc::Permutation>(m, "Permutation")
      .def("apply", py::overload_cast<const qc::Controls&>(
                        &qc::Permutation::apply, py::const_))
      .def("apply", py::overload_cast<const qc::Targets&>(
                        &qc::Permutation::apply, py::const_));

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
      .def_static("from_string",
                  [](const std::string& s) { return qc::opTypeFromString(s); });
}

} // namespace mqt
