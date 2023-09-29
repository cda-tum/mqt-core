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
#include <pybind11/stl_bind.h>
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
    .def(py::init<>(), "Constructs an empty QuantumComputation.")
      .def(py::init<std::size_t>(), "nq"_a,
           "Constructs an empty QuantumComputation with the given number of "
           "qubits.")
      .def(py::init<std::string>(), "filename"_a,
           "Read QuantumComputation from given file. Supported formats are "
           "[OpenQASM, Real, GRCS, TFC, QC]")
      .def_property("name", &qc::QuantumComputation::getName,
      &qc::QuantumComputation::setName)
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
    .def_readonly("initial_layout", &qc::QuantumComputation::initialLayout)
    .def_readonly("output_permutation", &qc::QuantumComputation::outputPermutation)
      .def_property("gphase", &qc::QuantumComputation::getGlobalPhase,
                    &qc::QuantumComputation::gphase)
    .def("i", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::i),
         "q"_a, "Apply the identity on qubit q.")
      .def("i", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                 &qc::QuantumComputation::i), "q"_a, "ctrl"_a, "Apply a controlled identity gate on qubit q with control ctrl.")
      .def("i", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                  &qc::QuantumComputation::i),
           "q"_a, "controls"_a,
           "Apply a multi-controlled identity gate on qubit q with controls controls.")
    .def("h", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::h),
         "q"_a, "Apply the Hadamard gate on qubit q.")
      .def("h", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                 &qc::QuantumComputation::h), "q"_a, "ctrl"_a, "Apply a controlled Hadamard gate on qubit q with control ctrl.")
      .def("h", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                  &qc::QuantumComputation::h),
           "q"_a, "controls"_a,
           "Apply a multi-controlled Hadamard gate on qubit q with controls controls.")
    .def("x", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::x),
         "q"_a, "Apply an X gate on qubit q.")
      .def("x", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                 &qc::QuantumComputation::x),
           "q"_a, "ctrl"_a,
           "Apply a controlled X gate on qubit q with control ctrl.")
      .def("x", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                  &qc::QuantumComputation::x),
           "q"_a, "controls"_a,
           "Apply a multi-controlled X gate on qubit q with controls controls.")
    .def("y", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::y),
         "q"_a, "Apply a Y gate on qubit q.")
      .def("y", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                 &qc::QuantumComputation::y),
           "q"_a, "ctrl"_a,
            "Apply a controlled Y gate on qubit q with control ctrl.")
      .def("y", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                  &qc::QuantumComputation::y),
           "q"_a, "controls"_a,
           "Apply a multi-controlled Y gate on qubit q with controls controls.")
    .def("z", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::z), "q"_a,
         "Apply a Z gate on qubit q.")
      .def("z", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                 &qc::QuantumComputation::z), "q"_a, "ctrl"_a, "Apply a controlled Z gate on qubit q with control ctrl.")
      .def("z", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                  &qc::QuantumComputation::z), "q"_a, "controls"_a, "Apply a multi-controlled Z gate on qubit q with controls controls.")
    .def("s", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::s), "q"_a,
         "Apply an S gate on qubit q.")
      .def("s", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                 &qc::QuantumComputation::s),
           "q"_a, "ctrl"_a,
           "Apply a controlled S gate on qubit q with control ctrl.")
      .def("s", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                  &qc::QuantumComputation::s),
           "q"_a, "controls"_a,
           "Apply a multi-controlled S gate on qubit q with controls controls.")
    .def("sdag", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::sdag),
         "q"_a, "Apply an Sdag gate on qubit q.")
      .def("sdag", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                    &qc::QuantumComputation::sdag),
           "q"_a, "ctrl"_a,
           "Apply a controlled Sdag gate on qubit q with control ctrl.")
      .def("sdag", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                     &qc::QuantumComputation::sdag),
           "q"_a, "controls"_a,
           "Apply a multi-controlled Sdag gate on qubit q with controls controls.")
    .def("t", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::t),
         "q"_a, "Apply a T gate on qubit q.")
      .def("t", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                 &qc::QuantumComputation::t),
           "q"_a, "ctrl"_a,
           "Apply a controlled T gate on qubit q with control ctrl.")
      .def("t", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                  &qc::QuantumComputation::t),
           "q"_a, "controls"_a,
           "Apply a multi-controlled T gate on qubit q with controls controls.")
    .def("tdag", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::tdag),
         "q"_a, "Apply a Tdag gate on qubit q.")
      .def("tdag", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                    &qc::QuantumComputation::tdag),
           "q"_a, "ctrl"_a,
           "Apply a controlled Tdag gate on qubit q with control ctrl.")
      .def("tdag", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                     &qc::QuantumComputation::tdag),
           "q"_a, "controls"_a,
           "Apply a multi-controlled Tdag gate on qubit q with controls controls.")
    .def("v", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::v),
         "q"_a, "Apply a V gate on qubit q.")
      .def("v", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                 &qc::QuantumComputation::v),
           "q"_a, "ctrl"_a,
           "Apply a controlled V gate on qubit q with control ctrl.")
      .def("v", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                  &qc::QuantumComputation::v),
           "q"_a, "controls"_a,
           "Apply a multi-controlled V gate on qubit q with controls controls.")
    .def("vdag", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::vdag),
         "q"_a, "Apply a Vdag gate on qubit q.")
      .def("vdag", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                    &qc::QuantumComputation::vdag),
           "q"_a, "ctrl"_a,
           "Apply a controlled Vdag gate on qubit q with control ctrl.")
      .def("vdag", py::overload_cast<qc::Qubit, const qc::Controls&>(
                       &qc::QuantumComputation::vdag)
           , "q"_a, "controls"_a,
           "Apply a multi-controlled Vdag gate on qubit q with controls controls.")
      .def("u3", py::overload_cast<qc::Qubit, const qc::fp, const qc::fp,
           const qc::fp>(&qc::QuantumComputation::u3),
           "q"_a, "theta"_a, "phi"_a, "lambda"_a,
           "Apply a U3 gate on qubit q with parameters theta, phi, lambda.")
      .def("u3", py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp,
                                   const qc::fp, const qc::fp>(
                                                               &qc::QuantumComputation::u3),
           "q"_a, "ctrl"_a, "theta"_a, "phi"_a, "lambda"_a,
           "Apply a controlled U3 gate on qubit q with control ctrl and parameters theta, phi, lambda.")
      .def("u3", py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp,
                                   const qc::fp, const qc::fp>(
                                                               &qc::QuantumComputation::u3),
           "q"_a, "controls"_a, "theta"_a, "phi"_a, "lambda"_a,
           "Apply a multi-controlled U3 gate on qubit q with controls controls and parameters theta, phi, lambda.")
      .def("u2", py::overload_cast<qc::Qubit, const qc::fp, const qc::fp>(
                                                                          &qc::QuantumComputation::u2),
           "q"_a, "phi"_a, "lambda"_a,
           "Apply a U2 gate on qubit q with parameters phi, lambda.")
      .def("u2", py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp,
           const qc::fp>(&qc::QuantumComputation::u2),
           "q"_a, "ctrl"_a, "phi"_a, "lambda"_a,
           "Apply a controlled U2 gate on qubit q with control ctrl and parameters phi, lambda.")
      .def("u2", py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp,
           const qc::fp>(&qc::QuantumComputation::u2),
           "q"_a, "controls"_a, "phi"_a, "lambda"_a,
           "Apply a multi-controlled U2 gate on qubit q with controls controls and parameters phi, lambda.")
      .def("phase", py::overload_cast<qc::Qubit, const qc::fp>(
                                                               &qc::QuantumComputation::phase),
           "q"_a, "lambda"_a,
           "Apply a phase gate on qubit q with parameter lambda.")
      .def("phase",
           py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
                                                                          &qc::QuantumComputation::phase),
           "q"_a, "ctrl"_a, "lambda"_a,
           "Apply a controlled phase gate on qubit q with control ctrl and parameter lambda.")
      .def("phase",
           py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
                                                                           &qc::QuantumComputation::phase),
           "q"_a, "controls"_a, "lambda"_a,
           "Apply a multi-controlled phase gate on qubit q with controls controls and parameter lambda.")
    .def("sx", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::sx),
         "q"_a, "Apply a square root of X gate on qubit q.")
      .def("sx", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                  &qc::QuantumComputation::sx),
           "q"_a, "ctrl"_a,
           "Apply a controlled square root of X gate on qubit q with control ctrl.")
      .def("sx", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                   &qc::QuantumComputation::sx),
           "q"_a, "controls"_a,
           "Apply a multi-controlled square root of X gate on qubit q with controls controls.")
      .def("sxdag",
           py::overload_cast<qc::Qubit>(&qc::QuantumComputation::sxdag),
           "q"_a, "Apply the inverse of the square root of X gate on qubit q."
           )
      .def("sxdag", py::overload_cast<qc::Qubit, const qc::Control&>(
                                                                     &qc::QuantumComputation::sxdag),
           "q"_a, "ctrl"_a,
           "Apply the controlled inverse of the square root of X gate on qubit q with control ctrl.")
      .def("sxdag", py::overload_cast<qc::Qubit, const qc::Controls&>(
                                                                      &qc::QuantumComputation::sxdag),
           "q"_a, "controls"_a,
           "Apply the multi-controlled inverse of the square root of X gate on qubit q with controls controls.")
      .def("rx", py::overload_cast<qc::Qubit, const qc::fp>(
                                                            &qc::QuantumComputation::rx),
           "q"_a, "theta"_a,
           "Apply an X-rotation gate on qubit q with angle theta.")
      .def("rx", py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
                                                                                &qc::QuantumComputation::rx),
           "q"_a, "ctrl"_a, "theta"_a,
           "Apply a controlled X-rotation gate on qubit q with control ctrl and angle theta.")
      .def("rx",
           py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
                                                                           &qc::QuantumComputation::rx),
           "q"_a, "controls"_a, "theta"_a,
           "Apply a multi-controlled X-rotation gate on qubit q with controls controls and angle theta.")
      .def("ry", py::overload_cast<qc::Qubit, const qc::fp>(
                                                            &qc::QuantumComputation::ry),
           "q"_a, "theta"_a,
           "Apply a Y-rotation gate on qubit q with angle theta.")
      .def("ry", py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
                                                                                &qc::QuantumComputation::ry),
           "q"_a, "ctrl"_a, "theta"_a,
           "Apply a controlled Y-rotation gate on qubit q with control ctrl and angle theta.")
      .def("ry",
           py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
                                                                           &qc::QuantumComputation::ry),
           "q"_a, "controls"_a, "theta"_a,
           "Apply a multi-controlled Y-rotation gate on qubit q with controls controls and angle theta.")
      .def("rz", py::overload_cast<qc::Qubit, const qc::fp>(
                                                            &qc::QuantumComputation::rz),
           "q"_a, "phi"_a,
           "Apply a Z-rotation gate on qubit q with angle phi.")
      .def("rz", py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
                                                                                &qc::QuantumComputation::rz),
           "q"_a, "ctrl"_a, "phi"_a,
           "Apply a controlled Z-rotation gate on qubit q with control ctrl and angle phi.")
      .def("rz",
           py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
                                                                           &qc::QuantumComputation::rz),
           "q"_a, "controls"_a, "phi"_a,
           "Apply a multi-controlled Z-rotation gate on qubit q with controls controls and angle phi.")
      .def("swap", py::overload_cast<qc::Qubit, qc::Qubit>(
                                                           &qc::QuantumComputation::swap),
           "q1"_a, "q2"_a,
           "Apply a SWAP gate on qubits q1 and q2.")
      .def("swap", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                                                                               &qc::QuantumComputation::swap),
           "q1"_a, "q2"_a, "ctrl"_a,
           "Apply a controlled SWAP (Fredkin) gate on qubits q1 and q2 with control ctrl.")
      .def("swap", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                                                                                &qc::QuantumComputation::swap),
           "q1"_a, "q2"_a, "controls"_a,
           "Apply a multi-controlled SWAP gate on qubits q1 and q2 with controls controls.")
      .def("iswap", py::overload_cast<qc::Qubit, qc::Qubit>(
                                                            &qc::QuantumComputation::iswap),
           "q1"_a, "q2"_a,
           "Apply an iSWAP gate on qubits q1 and q2.")
      .def("iswap", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                                                                                &qc::QuantumComputation::iswap),
           "q1"_a, "q2"_a, "ctrl"_a,
           "Apply a controlled iSWAP gate on qubits q1 and q2 with control ctrl.")
      .def("iswap",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                                                                        &qc::QuantumComputation::iswap),
           "q1"_a, "q2"_a, "controls"_a,
           "Apply a multi-controlled iSWAP gate on qubits q1 and q2 with controls controls.")
      .def("peres", py::overload_cast<qc::Qubit, qc::Qubit>(
                                                            &qc::QuantumComputation::peres),
           "q1"_a, "q2"_a,
           "Apply a Peres gate on qubits q1 and q2.")
      .def("peres", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                                                                                &qc::QuantumComputation::peres),
           "q1"_a, "q2"_a, "ctrl"_a,
           "Apply a controlled Peres gate on qubits q1 and q2 with control ctrl.")
      .def("peres",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                                                                        &qc::QuantumComputation::peres),
           "q1"_a, "q2"_a, "controls"_a,
           "Apply a multi-controlled Peres gate on qubits q1 and q2 with controls controls.")
      .def("peresdag", py::overload_cast<qc::Qubit, qc::Qubit>(
                                                               &qc::QuantumComputation::peresdag),
           "q1"_a, "q2"_a,
           "Apply an inverse Peres gate on qubits q1 and q2.")
      .def("peresdag",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                                                                       &qc::QuantumComputation::peresdag),
           "q1"_a, "q2"_a, "ctrl"_a,
           "Apply a controlled inverse Peres gate on qubits q1 and q2 with control ctrl.")
      .def("peresdag",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                                                                        &qc::QuantumComputation::peresdag),
           "q1"_a, "q2"_a, "controls"_a,
           "Apply a multi-controlled inverse Peres gate on qubits q1 and q2 with controls controls.")
      .def("dcx", py::overload_cast<qc::Qubit, qc::Qubit>(
                                                          &qc::QuantumComputation::dcx),
           "q1"_a, "q2"_a,
           "Apply a double CNOT gate on qubits q1 and q2.")
      .def("dcx", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                                                                              &qc::QuantumComputation::dcx),
           "q1"_a, "q2"_a, "ctrl"_a,
           "Apply a controlled double CNOT gate on qubits q1 and q2 with control ctrl.")
      .def("dcx", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                                                                               &qc::QuantumComputation::dcx),
           "q1"_a, "q2"_a, "controls"_a,
           "Apply a multi-controlled double CNOT gate on qubits q1 and q2 with controls controls.")
      .def("ecr", py::overload_cast<qc::Qubit, qc::Qubit>(
                                                          &qc::QuantumComputation::ecr),
           "q1"_a, "q2"_a,
           "Apply an echoed cross-resonance gate on qubits q1 and q2.")
      .def("ecr", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
                                                                              &qc::QuantumComputation::ecr),
           "q1"_a, "q2"_a, "ctrl"_a,
           "Apply a controlled echoed cross-resonance gate on qubits q1 and q2 with control ctrl.")
      .def("ecr", py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
                                                                               &qc::QuantumComputation::ecr),
           "q1"_a, "q2"_a, "controls"_a,
           "Apply a multi-controlled echoed cross-resonance gate on qubits q1 and q2 with controls controls.")
      .def("rxx", py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                                                                  &qc::QuantumComputation::rxx),
           "q1"_a, "q2"_a, "phi"_a,
           "Apply an XX-rotation gate on qubits q1 and q2 with angle phi.")
      .def("rxx",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
                                                                               &qc::QuantumComputation::rxx),
           "q1"_a, "q2"_a, "ctrl"_a, "phi"_a,
           "Apply a controlled XX-rotation gate on qubits q1 and q2 with control ctrl and angle phi.")
      .def("rxx",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
                                                                                &qc::QuantumComputation::rxx),
           "q1"_a, "q2"_a, "controls"_a, "phi"_a,
           "Apply a multi-controlled XX-rotation gate on qubits q1 and q2 with controls controls and angle phi.")
      .def("ryy", py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                                                                  &qc::QuantumComputation::ryy),
           "q1"_a, "q2"_a, "phi"_a,
           "Apply a YY-rotation gate on qubits q1 and q2 with angle phi.")
      .def("ryy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
                                                                               &qc::QuantumComputation::ryy),
           "q1"_a, "q2"_a, "ctrl"_a, "phi"_a,
           "Apply a controlled YY-rotation gate on qubits q1 and q2 with control ctrl and angle phi.")
      .def("ryy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
                                                                                &qc::QuantumComputation::ryy),
           "q1"_a, "q2"_a, "controls"_a, "phi"_a,
           "Apply a multi-controlled YY-rotation gate on qubits q1 and q2 with controls controls and angle phi.")
      .def("rzz", py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                                                                  &qc::QuantumComputation::rzz),
           "q1"_a, "q2"_a, "phi"_a,
           "Apply a ZZ-rotation gate on qubits q1 and q2 with angle phi.")
      .def("rzz",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
                                                                               &qc::QuantumComputation::rzz),
           "q1"_a, "q2"_a, "ctrl"_a, "phi"_a,
           "Apply a controlled ZZ-rotation gate on qubits q1 and q2 with control ctrl and angle phi.")
      .def("rzz",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
                                                                                &qc::QuantumComputation::rzz),
           "q1"_a, "q2"_a, "controls"_a, "phi"_a,
           "Apply a multi-controlled ZZ-rotation gate on qubits q1 and q2 with controls controls and angle phi.")
      .def("rzx", py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
                                                                  &qc::QuantumComputation::rzx),
           "q1"_a, "q2"_a, "phi"_a,
           "Apply a ZX-rotation gate on qubits q1 and q2 with angle phi.")
      .def("rzx",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
                                                                               &qc::QuantumComputation::rzx),
           "q1"_a, "q2"_a, "ctrl"_a, "phi"_a,
           "Apply a controlled ZX-rotation gate on qubits q1 and q2 with control ctrl and angle phi.")
      .def("rzx",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
                                                                                &qc::QuantumComputation::rzx),
           "q1"_a, "q2"_a, "controls"_a, "phi"_a,
           "Apply a multi-controlled ZX-rotation gate on qubits q1 and q2 with controls controls and angle phi.")
      .def("xx_minus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, qc::fp, qc::fp>(
                                                                   &qc::QuantumComputation::xx_minus_yy),
           "q1"_a, "q2"_a, "phi"_a, "lambda"_a,
           "Apply an XX-YY-rotation gate on qubits q1 and q2 with angles phi and lambda.")
      .def("xx_minus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp,
           qc::fp>(&qc::QuantumComputation::xx_minus_yy),
           "q1"_a, "q2"_a, "ctrl"_a, "phi"_a, "lambda"_a,
           "Apply a controlled XX-YY-rotation gate on qubits q1 and q2 with control ctrl and angles phi and lambda.")
      .def("xx_minus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp,
           qc::fp>(&qc::QuantumComputation::xx_minus_yy),
           "q1"_a, "q2"_a, "controls"_a, "phi"_a, "lambda"_a,
           "Apply a multi-controlled XX-YY-rotation gate on qubits q1 and q2 with controls controls and angles phi and lambda.")
      .def("xx_plus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, qc::fp, qc::fp>(
                                                                   &qc::QuantumComputation::xx_plus_yy),
           "q1"_a, "q2"_a, "phi"_a, "lambda"_a,
           "Apply an XX+YY-rotation gate on qubits q1 and q2 with angles phi and lambda.")
      .def("xx_plus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp,
           qc::fp>(&qc::QuantumComputation::xx_plus_yy),
           "q1"_a, "q2"_a, "ctrl"_a, "phi"_a, "lambda"_a,
           "Apply a controlled XX+YY-rotation gate on qubits q1 and q2 with control ctrl and angles phi and lambda.")
      .def("xx_plus_yy",
           py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp,
           qc::fp>(&qc::QuantumComputation::xx_plus_yy),
           "q1"_a, "q2"_a, "controls"_a, "phi"_a, "lambda"_a,
           "Apply a multi-controlled XX+YY-rotation gate on qubits q1 and q2 with controls controls and angles phi and lambda.")
      .def("measure", py::overload_cast<qc::Qubit, std::size_t>(
                                                                &qc::QuantumComputation::measure),
           "q"_a, "c"_a,
           "Measure qubit q and store the result in classical register c.")
      .def("measure",
           py::overload_cast<qc::Qubit, const std::pair<std::string, qc::Bit>&>(
                                                                                &qc::QuantumComputation::measure),
           "q"_a, "c"_a,
           "Measure qubit q and store the result in a named classical register c.")
      .def("measure", py::overload_cast<const std::vector<qc::Qubit>&,
                                        const std::vector<qc::Bit>&>(
                                                                     &qc::QuantumComputation::measure),
           "qs"_a, "cs"_a,
           "Measure qubits qs and store the result in classical register cs.")
      .def("reset",
           py::overload_cast<qc::Qubit>(&qc::QuantumComputation::reset),
           "q"_a,
           "Reset qubit q.")
      .def("reset", py::overload_cast<const std::vector<qc::Qubit>&>(
                                                                     &qc::QuantumComputation::reset),
           "qs"_a,
           "Reset qubits qs.")
      .def("barrier",
           py::overload_cast<qc::Qubit>(&qc::QuantumComputation::barrier),
           "q"_a,
           "Apply a barrier on qubit q.")
      .def("barrier", py::overload_cast<const std::vector<qc::Qubit>&>(
                                                                       &qc::QuantumComputation::barrier),
           "qs"_a,
           "Apply a barrier on qubits qs.")
      .def("classic_controlled",
           py::overload_cast<const qc::OpType, const qc::Qubit,
                             const qc::ClassicalRegister&, const std::uint64_t,
                             const std::vector<qc::fp>&>(
                                                         &qc::QuantumComputation::classicControlled),
           "op"_a, "q"_a, "c"_a, "t"_a, "params"_a,
           "Apply a classically controlled gate op on qubit q with classical control bit c and target qubit t.")
      .def("classic_controlled",
           py::overload_cast<const qc::OpType, const qc::Qubit,
                             const qc::Control, const qc::ClassicalRegister&,
                             const std::uint64_t, const std::vector<qc::fp>&>(
                                                                              &qc::QuantumComputation::classicControlled),
           "op"_a, "q"_a, "ctrl"_a, "c"_a, "t"_a, "params"_a,
           "Apply a classically controlled, parameterized gate op on qubit q with classical control bit c, target qubit t and parameters params.")
      .def("set_logical_qubit_ancillary",
           &qc::QuantumComputation::setLogicalQubitAncillary,
           "q"_a,
           "Set logical qubit q to be an ancillary qubit.")
    .def("add_qubit_register", &qc::QuantumComputation::addQubitRegister,
         "n"_a, "name"_a = "",
         "Add a register of n qubits with name name.")
    .def("add_classical_bit_register",
         &qc::QuantumComputation::addClassicalRegister,
         "n"_a, "name"_a = "",
         "Add a register of n classical bits with name name.")
      .def("add_ancillary_register",
           &qc::QuantumComputation::addAncillaryRegister,
           "n"_a, "name"_a = "",
           "Add a register of n ancillary qubits with name name.")
      .def("append_operation",
           [](qc::QuantumComputation& qc, const qc::Operation& op) {
             qc.emplace_back(op.clone()); // not an ideal solution but it works
           },
           "op"_a, "Append operation op to the quantum computation.") // Transfers ownership from Python to C++
    .def("instantiate", &qc::QuantumComputation::instantiate,
         "assignment"_a, "Instantiate the quantum computation by replacing all variables with their values dictated by the dict assignment which maps Variable objects to float.")
    .def("add_variable", &qc::QuantumComputation::addVariable,
         "var"_a, "Add variable var to the quantum computation.")
      .def("add_variables",
           [](qc::QuantumComputation& qc,
              const std::vector<qc::SymbolOrNumber>& vars) {
             for (const auto& var : vars) {
               qc.addVariable(var);
             }
           }, "vars"_a,
           "Add variables vars to the quantum computation.")
    .def("is_variable_free", &qc::QuantumComputation::isVariableFree,
         "Check if the quantum computation is free of variables.")
    .def("get_variables", &qc::QuantumComputation::getVariables, "Get all variables used in the quantum computation.")
      .def("initialize_io_mapping",
           &qc::QuantumComputation::initializeIOMapping, "Initialize the I/O mapping of the quantum computation."
           "If no initial mapping is given, the identity mapping will be assumed."
           "If no output permutation is given, it is derived from the measurements")
      .def("from_file", py::overload_cast<const std::string&>(
                                                              &qc::QuantumComputation::import),
           "filename"_a,
           "Import the quantum computation from file."
           "Supported formats are:"
           " - OpenQASM 2.0 (.qasm)"
           " - Real (.real)"
           " - GRCS (.grcs)"
           " - TFC (.tfc)"
           " - QC (.qc)")
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
           },
           "filename"_a, "format"_a,
           "Import the quantum computation from file with specified format."
           "Supported formats are:"
           " - OpenQASM 2.0 (.qasm)"
            " - Real (.real)"
            " - GRCS (.grcs)"
            " - TFC (.tfc)"
           " - qc (.qc)")
      .def("dump",
           py::overload_cast<const std::string&>(&qc::QuantumComputation::dump),
           "filename"_a, "Dump the quantum computation to file."
           "Supported formats are:"
           " - OpenQASM 2.0 (.qasm)"
            " - Real (.real)"
            " - GRCS (.grcs)"
            " - TFC (.tfc)"
            " - qc (.qc)"
            " - Tensor (.tensor)")
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
           },
           "filename"_a, "format"_a,
           "Dump the quantum computation to file with specified format."
                      "Supported formats are:"
           " - OpenQASM 2.0 (.qasm)"
            " - Real (.real)"
            " - GRCS (.grcs)"
            " - TFC (.tfc)"
            " - qc (.qc)"
            " - Tensor (.tensor)")
      .def("to_open_qasm",
           [](qc::QuantumComputation& qc) {
             std::ostringstream oss;
             qc.dumpOpenQASM(oss);
             return oss.str();
           },
           "Dump the quantum computation to a string in OpenQASM 2.0 format.")
           .def("__len__", &qc::QuantumComputation::getNindividualOps,
                "Get the number of operations in the quantum computation.")
      .def("__getitem__", [](const qc::QuantumComputation& qc,
                             std::size_t idx) { return qc.at(idx).get(); }, py::return_value_policy::reference_internal);

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
      .def("size", &qc::CompoundOperation::size,
           "Return number of sub-operations.")
      .def("empty", &qc::CompoundOperation::empty)
      .def("__getitem__", [](const qc::CompoundOperation& op,
                             std::size_t i) { return op.at(i).get(); }, py::return_value_policy::reference_internal)
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

  py::class_<sym::Variable>(m, "Variable", "A symbolic variable.")
      .def(py::init<std::string>(), "name"_a = "",
           "Create a variable with a given variable name. Variables are "
           "uniquely identified by their name, so if a variable with the same "
           "name already exists, the existing variable will be returned.")
      .def_property_readonly("name", &sym::Variable::getName)
      .def("__str__", &sym::Variable::getName)
      .def("__eq__", &sym::Variable::operator==)
      .def("__ne__", &sym::Variable::operator!=)
      .def("__lt__", &sym::Variable::operator<)
      .def("__gt__", &sym::Variable::operator>);

  py::class_<sym::Term<double>>(
      m, "Term",
      "A symbolic term which consists of a variable with a given coefficient.")
      .def(py::init<double, sym::Variable>(), "coefficient"_a, "variable"_a,
           "Create a term with a given coefficient and variable.")
      .def(py::init<sym::Variable>(), "variable"_a,
           "Create a term with a given variable and coefficient 1.")
      .def_property_readonly("variable", &sym::Term<double>::getVar,
                             "Return the variable of this term.")
      .def_property_readonly("coefficient", &sym::Term<double>::getCoeff,
                             "Return the coefficient of this term.")
      .def("has_zero_coefficient", &sym::Term<double>::hasZeroCoeff,
           "Return true if the coefficient of this term is zero.")
      .def("add_coefficient", &sym::Term<double>::addCoeff, "coeff"_a,
           "Add coeff a to the coefficient of this term.")
      .def("evaluate", &sym::Term<double>::evaluate, "assignment"_a,
           "Return the value of this term given by multiplying the coefficient "
           "of this term to the variable value dictated by the assignment.")
      .def("__mul__",
           [](const sym::Term<double>& lhs, double rhs) { return lhs * rhs; })
      .def("__rmul__",
           [](sym::Term<double> rhs, const double lhs) { return lhs * rhs; })
      .def("__truediv__",
           [](const sym::Term<double>& lhs, double rhs) { return lhs / rhs; })
      .def("__rtruediv__",
           [](sym::Term<double> rhs, const double lhs) { return lhs / rhs; });

  py::class_<sym::Expression<double, double>>(
      m, "Expression",
      "Class representing a symbolic sum of terms. The expression is of the "
      "form `constant + term_1 + term_2 + ... + term_n`.")
      .def(py::init<>(), "Create an empty expression.")
      .def(
          "__init__",
          [](sym::Expression<double, double>* expr,
             const std::vector<sym::Term<double>>& terms, double constant) {
            new (expr) sym::Expression<double, double>(terms, constant);
          },
          "terms"_a, "constant"_a = 0.0,
          "Create an expression with a given list of terms and a constant (0 "
          "by default).")
      .def(
          "__init__",
          [](sym::Expression<double, double>* expr,
             const sym::Term<double>& term, double constant) {
            new (expr) sym::Expression<double, double>(
                std::vector<sym::Term<double>>{term}, constant);
          },
          "term"_a, "constant"_a = 0.0,
          "Create an expression with a given term and a constant (0 by "
          "default).")
      .def(py::init<double>(), "constant"_a,
           "Create a constant expression involving no symbolic terms.")
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

      .def("is_zero", &sym::Expression<double, double>::isZero,
           "Return true if this expression is zero, i.e., all terms have "
           "coefficient 0 and the constant is 0 as well.")
      .def("is_constant", &sym::Expression<double, double>::isConstant,
           "Return true if this expression is constant, i.e., all terms have "
           "coefficient 0 or no terms are involved.")
      .def("num_terms", &sym::Expression<double, double>::numTerms,
           "Return the number of terms in this expression.")
      .def("__len__", &sym::Expression<double, double>::numTerms)
      .def_property_readonly("terms",
                             &sym::Expression<double, double>::getTerms)
      .def("evaluate", &sym::Expression<double, double>::evaluate,
           "assignment"_a,
           "Return the value of this expression given by summing the values of "
           "all instantiated terms and the constant given by the assignment.")
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

  py::enum_<qc::OpType>(m, "OpType",
                        "Enum class for representing quantum operations.")
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
      .def(py::init([](const std::string& str) -> qc::OpType {
        return qc::opTypeFromString(str);
      }));
  py::implicitly_convertible<py::str, qc::OpType>();
}

} // namespace mqt
