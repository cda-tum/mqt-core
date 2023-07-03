#include "Definitions.hpp"
#include "Permutation.hpp"
#include "QuantumComputation.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"
#include "operations/StandardOperation.hpp"

#include <cstddef>
#include <iostream>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <ostream>
#include <python/nanobind.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace mqt {
// void register_qiskit(nb::module_);

enum class foo { a };

foo fooFromStr(const std::string& s) { return foo::a; }
NB_MODULE(_core, m) {

  nb::class_<qc::QuantumComputation>(
      m, "QuantumComputation",
      "Representation of quantum circuits within MQT Core")
      .def(nb::init<std::size_t>(), "nq"_a,
           "Constructs an empty QuantumComputation with the given number of "
           "qubits.")
      .def(nb::init<std::string>(), "filename"_a,
           "Read QuantumComputation from given file. Supported formats are "
           "[OpenQASM, Real, GRCS, TFC, QC]")
      // .def_rw("name", &qc::QuantumComputation::getName,
      // &qc::QuantumComputation::setName)
      .def("clone", &qc::QuantumComputation::clone,
           "Clone this QuantumComputation object.")
      .def_prop_ro("n_qubits", &qc::QuantumComputation::getNqubits)
      .def_prop_ro("n_ancillae", &qc::QuantumComputation::getNancillae)
      .def_prop_ro("n_qubits_without_ancillae",
                   &qc::QuantumComputation::getNqubitsWithoutAncillae)
      .def_prop_ro("n_cbits", &qc::QuantumComputation::getNcbits)
      .def_prop_ro("n_ops", &qc::QuantumComputation::getNops)
      .def_prop_ro("n_single_qubit_ops",
                   &qc::QuantumComputation::getNsingleQubitOps)
      .def_prop_ro("n_individual_ops",
                   &qc::QuantumComputation::getNindividualOps)
      .def_prop_ro("depth", &qc::QuantumComputation::getDepth)
      .def_prop_rw("gphase", &qc::QuantumComputation::getGlobalPhase,
                   &qc::QuantumComputation::gphase)
      .def("i", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::i))
      .def("i", nb::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::i))
      .def("i", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::i))
      .def("h", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::h))
      .def("h", nb::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::h))
      .def("h", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::h))
      .def("x", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::x))
      .def("x", nb::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::x))
      .def("x", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::x))
      .def("y", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::y))
      .def("y", nb::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::y))
      .def("y", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::y))
      .def("z", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::z))
      .def("z", nb::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::z))
      .def("z", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::z))
      .def("s", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::s))
      .def("s", nb::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::s))
      .def("s", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::s))
      .def("sdag", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::sdag))
      .def("sdag", nb::overload_cast<qc::Qubit, const qc::Control&>(
                       &qc::QuantumComputation::sdag))
      .def("sdag", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                       &qc::QuantumComputation::sdag))
      .def("t", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::t))
      .def("t", nb::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::t))
      .def("t", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::t))
      .def("tdag", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::tdag))
      .def("tdag", nb::overload_cast<qc::Qubit, const qc::Control&>(
                       &qc::QuantumComputation::tdag))
      .def("tdag", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                       &qc::QuantumComputation::tdag))
      .def("v", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::v))
      .def("v", nb::overload_cast<qc::Qubit, const qc::Control&>(
                    &qc::QuantumComputation::v))
      .def("v", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                    &qc::QuantumComputation::v))
      .def("vdag", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::vdag))
      .def("vdag", nb::overload_cast<qc::Qubit, const qc::Control&>(
                       &qc::QuantumComputation::vdag))
      .def("vdag", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                       &qc::QuantumComputation::vdag))
      .def("u3", nb::overload_cast<qc::Qubit, const qc::fp, const qc::fp,
                                   const qc::fp>(&qc::QuantumComputation::u3))
      .def("u3", nb::overload_cast<qc::Qubit, const qc::Control&, const qc::fp,
                                   const qc::fp, const qc::fp>(
                     &qc::QuantumComputation::u3))
      .def("u3", nb::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp,
                                   const qc::fp, const qc::fp>(
                     &qc::QuantumComputation::u3))
      .def("u2", nb::overload_cast<qc::Qubit, const qc::fp, const qc::fp>(
                     &qc::QuantumComputation::u2))
      .def("u2", nb::overload_cast<qc::Qubit, const qc::Control&, const qc::fp,
                                   const qc::fp>(&qc::QuantumComputation::u2))
      .def("u2", nb::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp,
                                   const qc::fp>(&qc::QuantumComputation::u2))
      .def("phase", nb::overload_cast<qc::Qubit, const qc::fp>(
                        &qc::QuantumComputation::phase))
      .def("phase",
           nb::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
               &qc::QuantumComputation::phase))
      .def("phase",
           nb::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
               &qc::QuantumComputation::phase))
      .def("sx", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::sx))
      .def("sx", nb::overload_cast<qc::Qubit, const qc::Control&>(
                     &qc::QuantumComputation::sx))
      .def("sx", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                     &qc::QuantumComputation::sx))
      .def("sxdag",
           nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::sxdag))
      .def("sxdag", nb::overload_cast<qc::Qubit, const qc::Control&>(
                        &qc::QuantumComputation::sxdag))
      .def("sxdag", nb::overload_cast<qc::Qubit, const qc::Controls&>(
                        &qc::QuantumComputation::sxdag))
      .def("rx", nb::overload_cast<qc::Qubit, const qc::fp>(
                     &qc::QuantumComputation::rx))
      .def("rx", nb::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
                     &qc::QuantumComputation::rx))
      .def("rx",
           nb::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
               &qc::QuantumComputation::rx))
      .def("ry", nb::overload_cast<qc::Qubit, const qc::fp>(
                     &qc::QuantumComputation::ry))
      .def("ry", nb::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
                     &qc::QuantumComputation::ry))
      .def("ry",
           nb::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
               &qc::QuantumComputation::ry))
      .def("rz", nb::overload_cast<qc::Qubit, const qc::fp>(
                     &qc::QuantumComputation::rz))
      .def("rz", nb::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
                     &qc::QuantumComputation::rz))
      .def("rz",
           nb::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
               &qc::QuantumComputation::rz))
      .def("swap", nb::overload_cast<qc::Qubit, qc::Qubit>(
                       &qc::QuantumComputation::swap))
      .def("swap", nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                       &qc::QuantumComputation::swap))
      .def("swap", nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                       &qc::QuantumComputation::swap))
      .def("iswap", nb::overload_cast<qc::Qubit, qc::Qubit>(
                        &qc::QuantumComputation::iswap))
      .def("iswap", nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                        &qc::QuantumComputation::iswap))
      .def("iswap",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
               &qc::QuantumComputation::iswap))
      .def("peres", nb::overload_cast<qc::Qubit, qc::Qubit>(
                        &qc::QuantumComputation::peres))
      .def("peres", nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                        &qc::QuantumComputation::peres))
      .def("peres",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
               &qc::QuantumComputation::peres))
      .def("peresdag", nb::overload_cast<qc::Qubit, qc::Qubit>(
                           &qc::QuantumComputation::peresdag))
      .def("peresdag",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
               &qc::QuantumComputation::peresdag))
      .def("peresdag",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
               &qc::QuantumComputation::peresdag))
      .def("dcx", nb::overload_cast<qc::Qubit, qc::Qubit>(
                      &qc::QuantumComputation::dcx))
      .def("dcx", nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                      &qc::QuantumComputation::dcx))
      .def("dcx", nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                      &qc::QuantumComputation::dcx))
      .def("ecr", nb::overload_cast<qc::Qubit, qc::Qubit>(
                      &qc::QuantumComputation::ecr))
      .def("ecr", nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                      &qc::QuantumComputation::ecr))
      .def("ecr", nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                      &qc::QuantumComputation::ecr))
      .def("rxx", nb::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                      &qc::QuantumComputation::rxx))
      .def("rxx",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
               &qc::QuantumComputation::rxx))
      .def("rxx",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
               &qc::QuantumComputation::rxx))
      .def("ryy", nb::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                      &qc::QuantumComputation::ryy))
      .def("ryy",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
               &qc::QuantumComputation::ryy))
      .def("ryy",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
               &qc::QuantumComputation::ryy))
      .def("rzz", nb::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                      &qc::QuantumComputation::rzz))
      .def("rzz",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
               &qc::QuantumComputation::rzz))
      .def("rzz",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
               &qc::QuantumComputation::rzz))
      .def("rzx", nb::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                      &qc::QuantumComputation::rzx))
      .def("rzx",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
               &qc::QuantumComputation::rzx))
      .def("rzx",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
               &qc::QuantumComputation::rzx))
      .def("xx_minus_yy",
           nb::overload_cast<qc::Qubit, qc::Qubit, qc::fp, qc::fp>(
               &qc::QuantumComputation::xx_minus_yy))
      .def("xx_minus_yy",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp,
                             qc::fp>(&qc::QuantumComputation::xx_minus_yy))
      .def("xx_minus_yy",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp,
                             qc::fp>(&qc::QuantumComputation::xx_minus_yy))
      .def("xx_plus_yy",
           nb::overload_cast<qc::Qubit, qc::Qubit, qc::fp, qc::fp>(
               &qc::QuantumComputation::xx_plus_yy))
      .def("xx_plus_yy",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp,
                             qc::fp>(&qc::QuantumComputation::xx_plus_yy))
      .def("xx_plus_yy",
           nb::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp,
                             qc::fp>(&qc::QuantumComputation::xx_plus_yy))
      .def("measure", nb::overload_cast<qc::Qubit, std::size_t>(
                          &qc::QuantumComputation::measure))
      .def("measure",
           nb::overload_cast<qc::Qubit, const std::pair<std::string, qc::Bit>&>(
               &qc::QuantumComputation::measure))
      .def("measure", nb::overload_cast<const std::vector<qc::Qubit>&,
                                        const std::vector<qc::Bit>&>(
                          &qc::QuantumComputation::measure))
      .def("reset",
           nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::reset))
      .def("reset", nb::overload_cast<const std::vector<qc::Qubit>&>(
                        &qc::QuantumComputation::reset))
      .def("barrier",
           nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::barrier))
      .def("barrier", nb::overload_cast<const std::vector<qc::Qubit>&>(
                          &qc::QuantumComputation::barrier))
      .def("reset",
           nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::reset))
      .def("classic_controlled",
           nb::overload_cast<const qc::OpType, const qc::Qubit,
                             const qc::ClassicalRegister&, const std::uint64_t,
                             const std::vector<qc::fp>&>(
               &qc::QuantumComputation::classicControlled))
      .def("classic_controlled",
           nb::overload_cast<const qc::OpType, const qc::Qubit,
                             const qc::Control, const qc::ClassicalRegister&,
                             const std::uint64_t, const std::vector<qc::fp>&>(
               &qc::QuantumComputation::classicControlled))
      .def("classic_controlled",
           nb::overload_cast<const qc::OpType, const qc::Qubit,
                             const qc::Controls&, const qc::ClassicalRegister&,
                             const std::uint64_t, const std::vector<qc::fp>&>(
               &qc::QuantumComputation::classicControlled))
      // .def("__str__", [](const qc::QuantumComputation& qc){std::stringstream
      // ss;qc.print(ss);return ss.str();})
      // .def("__iter__", [](const qc::QuantumComputation& qc) {
      //   return nb::make_iterator(nb::type<qc::QuantumComputation>(), "ops",
      //   qc.begin(), qc.end());
      // }, nb::keep_alive<0, 1>())
      .def("__len__", &qc::QuantumComputation::getNindividualOps)
      .def(
          "__getitem__",
          [](const qc::QuantumComputation& qc,
             std::size_t
                 idx) { return qc.at(idx).get(); }, // Beware: this gives write
                                                    // access to underlying
                                                    // Operation
          nb::rv_policy::reference)
      // .def_prop_ro("ops", [](const qc::QuantumComputation& qc, std::size_t
      // idx){return *qc.at(idx);})
      ;
  nb::class_<qc::Control>(m, "Control")
      .def_rw("type", &qc::Control::type)
      .def_rw("qubit", &qc::Control::qubit)
      .def(nb::init<qc::Qubit>())
      .def(nb::init<qc::Qubit, qc::Control::Type>());

  nb::enum_<qc::Control::Type>(m, "ControlType")
      .value("Pos", qc::Control::Type::Pos)
      .value("Neg", qc::Control::Type::Neg)
      .export_values();

  nb::class_<qc::Operation>(m, "Operation")
      // .def(nb::init<>())
      // .def_prop_rw("targets", &qc::Operation::getTargets,
      // &qc::Operation::setTargets)
      .def_prop_rw("name", &qc::Operation::getName, &qc::Operation::setName)
      .def("set_gate", &qc::Operation::setGate);
  ;
  nb::class_<qc::StandardOperation, qc::Operation>(m, "StandardOperation")
      .def(nb::init<>())
      .def(nb::init<std::size_t, qc::Qubit, qc::OpType, std::vector<qc::fp>,
                    qc::Qubit>(),
           "nq"_a, "target"_a, "op_type"_a, "params"_a = std::vector<qc::fp>{},
           "starting_qubit"_a = 0)
      .def(nb::init<std::size_t, const qc::Targets&, qc::OpType,
                    std::vector<qc::fp>, qc::Qubit>(),
           "nq"_a, "targets"_a, "op_type"_a, "params"_a = std::vector<qc::fp>{},
           "starting_qubit"_a = 0)
      .def(nb::init<std::size_t, qc::Control, qc::Qubit, qc::OpType,
                    const std::vector<qc::fp>&, qc::Qubit>(),
           "nq"_a, "control"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0)
      .def(nb::init<std::size_t, qc::Control, const qc::Targets&, qc::OpType,
                    const std::vector<qc::fp>&, qc::Qubit>(),
           "nq"_a, "control"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0)
      .def(nb::init<std::size_t, const qc::Controls&, qc::Qubit, qc::OpType,
                    const std::vector<qc::fp>&, qc::Qubit>(),
           "nq"_a, "controls"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0)
      .def(nb::init<std::size_t, const qc::Controls&, const qc::Targets&,
                    qc::OpType, std::vector<qc::fp>, qc::Qubit>(),
           "nq"_a, "controls"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0)
      .def(nb::init<std::size_t, const qc::Controls&, qc::Qubit, qc::Qubit>(),
           "nq"_a, "controls"_a, "target"_a, "starting_qubit"_a = 0)
      .def(nb::init<std::size_t, const qc::Controls&, qc::Qubit, qc::Qubit,
                    qc::OpType, std::vector<qc::fp>, qc::Qubit>(),
           "nq"_a, "controls"_a, "target0"_a, "target1"_a, "op_type"_a,
           "params"_a = std::vector<qc::fp>{}, "starting_qubit"_a = 0)
      .def("is_standard_operation", &qc::StandardOperation::isStandardOperation)
      .def("clone", &qc::StandardOperation::clone)
      .def("equals", nb::overload_cast<const qc::Operation&>(
                         &qc::StandardOperation::equals, nb::const_))
      .def("equals",
           nb::overload_cast<const qc::Operation&, const qc::Permutation&,
                             const qc::Permutation&>(
               &qc::StandardOperation::equals, nb::const_))
      .def("__str__", [](const qc::StandardOperation& qc) {
        std::ostringstream ss;
        qc.dumpOpenQASM(ss, {{"q", "0"}, {"q", "1"}, {"q", "2"}}, {});
        ss.str();
      });

  nb::class_<qc::Permutation>(m, "Permutation")
      .def("apply", nb::overload_cast<const qc::Controls&>(
                        &qc::Permutation::apply, nb::const_))
      .def("apply", nb::overload_cast<const qc::Targets&>(
                        &qc::Permutation::apply, nb::const_))
      .def(nb::init_implicit<std::map<
               qc::Qubit, qc::Qubit>>()); // Allows for implicit conversion from
                                          // dict[int, int] to Permutation

  nb::enum_<qc::OpType>(m, "OpType")
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
      .value("snapshot", qc::OpType::Snapshot)
      .value("showprobabilities", qc::OpType::ShowProbabilities)
      .value("barrier", qc::OpType::Barrier)
      .value("teleportation", qc::OpType::Teleportation)
      .value("classiccontrolled", qc::OpType::ClassicControlled)
      .export_values()
      .def_static("from_string",
                  [](const std::string& s) { return qc::opTypeFromString(s); });
}

} // namespace mqt
