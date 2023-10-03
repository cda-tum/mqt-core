#include "QuantumComputation.hpp"
#include "operations/Control.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"
#include "python/pybind11.hpp"

namespace mqt {

using DiffType = std::vector<std::unique_ptr<qc::Operation>>::difference_type;
using SizeType = std::vector<std::unique_ptr<qc::Operation>>::size_type;

void registerQuantumComputation(py::module& m) {
  auto wrap = [](DiffType i, const SizeType size) {
    if (i < 0) {
      i += static_cast<DiffType>(size);
    }
    if (i < 0 || static_cast<SizeType>(i) >= size) {
      throw py::index_error();
    }
    return i;
  };

  auto qc = py::class_<qc::QuantumComputation>(
      m, "QuantumComputation",
      "Representation of quantum circuits within MQT Core");

  ///---------------------------------------------------------------------------
  ///                           \n Constructors \n
  ///---------------------------------------------------------------------------

  qc.def(py::init<>(), "Constructs an empty QuantumComputation.");
  qc.def(py::init<std::size_t>(), "nq"_a,
         "Constructs an empty QuantumComputation with the given number of "
         "qubits.");
  qc.def(py::init<std::string>(), "filename"_a,
         "Read QuantumComputation from given file. Supported formats are "
         "[OpenQASM2, Real, GRCS, TFC, QC]");

  ///---------------------------------------------------------------------------
  ///                       \n General Properties \n
  ///---------------------------------------------------------------------------

  qc.def_property("name", &qc::QuantumComputation::getName,
                  &qc::QuantumComputation::setName);
  qc.def_property_readonly("num_qubits", &qc::QuantumComputation::getNqubits);
  qc.def_property_readonly("num_ancilla_qubits",
                           &qc::QuantumComputation::getNancillae);
  qc.def_property_readonly("num_qubits_without_ancilla_qubits",
                           &qc::QuantumComputation::getNqubitsWithoutAncillae);
  qc.def_property_readonly("num_classical_bits",
                           &qc::QuantumComputation::getNcbits);
  qc.def_property_readonly("num_ops", &qc::QuantumComputation::getNops);
  qc.def("num_single_qubit_ops", &qc::QuantumComputation::getNsingleQubitOps,
         "Returns the number of single-qubit operations.");
  qc.def("num_total_ops", &qc::QuantumComputation::getNindividualOps,
         "Returns the number of total operations by recursively counting "
         "sub-operations.");
  qc.def("depth", &qc::QuantumComputation::getDepth,
         "Returns the depth of the circuit.");
  qc.def_property("global_phase", &qc::QuantumComputation::getGlobalPhase,
                  &qc::QuantumComputation::gphase);
  qc.def("invert", &qc::QuantumComputation::invert,
         "Invert the quantum computation by inverting all operations and "
         "reversing the order of the operations.");
  qc.def("to_operation", &qc::QuantumComputation::asOperation,
         "Convert the quantum computation to a single operation. This gives "
         "ownership of the quantum computation to the returned operation.");

  ///---------------------------------------------------------------------------
  ///                  \n Mutable Sequence Interface \n
  ///---------------------------------------------------------------------------

  qc.def(
      "__getitem__",
      [&wrap](const qc::QuantumComputation& circ, DiffType i) {
        i = wrap(i, circ.getNops());
        return circ.at(static_cast<SizeType>(i)).get();
      },
      py::return_value_policy::reference_internal, "idx"_a,
      "Get the operation at index idx. Beware: this gives write access to "
      "the operation.");
  qc.def(
      "__getitem__",
      [](qc::QuantumComputation& circ, const py::slice& slice) {
        std::size_t start{};
        std::size_t stop{};
        std::size_t step{};
        std::size_t sliceLength{};
        if (!slice.compute(circ.getNops(), &start, &stop, &step,
                           &sliceLength)) {
          throw py::error_already_set();
        }
        auto ops = std::vector<qc::Operation*>();
        ops.reserve(sliceLength);
        for (std::size_t i = start; i < stop; i += step) {
          ops.emplace_back(circ.at(i).get());
        }
        return ops;
      },
      py::return_value_policy::reference_internal, "slice"_a,
      "Get a slice of the quantum computation. Beware: this gives write "
      "access to the operations.");
  qc.def(
      "__setitem__",
      [&wrap](qc::QuantumComputation& circ, DiffType i,
              const qc::Operation& op) {
        i = wrap(i, circ.getNops());
        circ.at(static_cast<SizeType>(i)) = op.clone();
      },
      "idx"_a, "op"_a, "Set the operation at index idx to op.");
  qc.def(
      "__setitem__",
      [](qc::QuantumComputation& circ, const py::slice& slice,
         const std::vector<qc::Operation*>& ops) {
        std::size_t start{};
        std::size_t stop{};
        std::size_t step{};
        std::size_t sliceLength{};
        if (!slice.compute(circ.getNops(), &start, &stop, &step,
                           &sliceLength)) {
          throw py::error_already_set();
        }
        if (sliceLength != ops.size()) {
          throw std::runtime_error(
              "Length of slice and number of operations do not match.");
        }
        for (std::size_t i = 0; i < sliceLength; ++i) {
          circ.at(start) = ops[i]->clone();
          start += step;
        }
      },
      "slice"_a, "ops"_a, "Set a slice of the quantum computation to ops.");
  qc.def(
      "__delitem__",
      [&wrap](qc::QuantumComputation& circ, DiffType i) {
        i = wrap(i, circ.getNops());
        circ.erase(circ.begin() + i);
      },
      "idx"_a, "Delete the operation at index idx.");
  qc.def(
      "__delitem__",
      [](qc::QuantumComputation& circ, const py::slice& slice) {
        std::size_t start{};
        std::size_t stop{};
        std::size_t step{};
        std::size_t sliceLength{};
        if (!slice.compute(circ.getNops(), &start, &stop, &step,
                           &sliceLength)) {
          throw py::error_already_set();
        }
        // delete in reverse order to not invalidate indices
        for (std::size_t i = sliceLength; i > 0; --i) {
          circ.erase(circ.begin() +
                     static_cast<int64_t>(start + (i - 1) * step));
        }
      },
      "slice"_a, "Delete a slice of the quantum computation.");
  qc.def("__len__", &qc::QuantumComputation::getNops,
         "Get the number of operations in the quantum computation.");
  qc.def(
      "insert",
      [](qc::QuantumComputation& circ, std::size_t idx,
         const qc::Operation& op) {
        circ.insert(circ.begin() + static_cast<int64_t>(idx), op.clone());
      },
      "idx"_a, "op"_a, "Insert op at index idx.");
  qc.def(
      "append",
      [](qc::QuantumComputation& circ, const qc::Operation& op) {
        circ.emplace_back(op.clone());
      },
      "op"_a, "Append an operation to the quantum computation.");
  qc.def("reverse", &qc::QuantumComputation::reverse,
         "Reverse the quantum sequence of operations.");
  qc.def("clear", py::overload_cast<>(&qc::QuantumComputation::reset),
         "Completely reset the quantum computation.");

  ///---------------------------------------------------------------------------
  ///                         \n (Qu)Bit Registers \n
  ///---------------------------------------------------------------------------

  qc.def("add_qubit_register", &qc::QuantumComputation::addQubitRegister, "n"_a,
         "name"_a = "q", "Add a register of `n` qubits with name `name`.");
  qc.def("add_classical_register",
         &qc::QuantumComputation::addClassicalRegister, "n"_a, "name"_a = "c",
         "Add a register of `n` classical bits with name `name`.");
  qc.def("add_ancillary_register",
         &qc::QuantumComputation::addAncillaryRegister, "n"_a, "name"_a = "anc",
         "Add a register of n ancillary qubits with name `name`.");
  qc.def("unify_quantum_registers",
         &qc::QuantumComputation::unifyQuantumRegisters, "name"_a = "q",
         "Unify all quantum registers into a single register.");

  ///---------------------------------------------------------------------------
  ///               \n Input Layout and Output Permutation \n
  ///---------------------------------------------------------------------------

  qc.def_readwrite("initial_layout", &qc::QuantumComputation::initialLayout);
  qc.def_readwrite("output_permutation",
                   &qc::QuantumComputation::outputPermutation);
  qc.def(
      "initialize_io_mapping", &qc::QuantumComputation::initializeIOMapping,
      "Initialize the I/O mapping of the quantum computation."
      "If no initial layout was previously set, the identity mapping will be "
      "assumed."
      "If the circuit contains measurements at the end, these will be "
      "used to deduce the output permutation.");

  ///---------------------------------------------------------------------------
  ///                  \n Ancillary and Garbage Handling \n
  ///---------------------------------------------------------------------------

  qc.def_readonly("ancillary", &qc::QuantumComputation::ancillary);
  qc.def("set_circuit_qubit_ancillary",
         &qc::QuantumComputation::setLogicalQubitAncillary, "q"_a,
         "Set the circuit's (logical) qubit q to be an ancillary qubit.");
  qc.def("is_circuit_qubit_ancillary",
         &qc::QuantumComputation::logicalQubitIsAncillary, "q"_a,
         "Check if the circuit's (logical) qubit q is an ancillary qubit.");
  qc.def_readonly("garbage", &qc::QuantumComputation::garbage);
  qc.def("set_circuit_qubit_garbage",
         &qc::QuantumComputation::setLogicalQubitGarbage, "q"_a,
         "Set the circuit's (logical) qubit q to be a garbage output.");
  qc.def("is_circuit_qubit_garbage",
         &qc::QuantumComputation::logicalQubitIsGarbage, "q"_a,
         "Check if the circuit's (logical) qubit q is a garbage output.");

  ///---------------------------------------------------------------------------
  ///                    \n Symbolic Circuit Handling \n
  ///---------------------------------------------------------------------------

  qc.def_property_readonly(
      "variables", &qc::QuantumComputation::getVariables,
      "Get all variables used in the quantum computation.");
  qc.def("add_variable", &qc::QuantumComputation::addVariable, "var"_a,
         "Add variable var to the quantum computation.");
  qc.def(
      "add_variables",
      [](qc::QuantumComputation& circ,
         const std::vector<qc::SymbolOrNumber>& vars) {
        for (const auto& var : vars) {
          circ.addVariable(var);
        }
      },
      "vars_"_a, "Add variables vars to the quantum computation.");
  qc.def("is_variable_free", &qc::QuantumComputation::isVariableFree,
         "Check if the quantum computation is free of variables.");
  qc.def("instantiate", &qc::QuantumComputation::instantiate, "assignment"_a,
         "Instantiate the quantum computation by replacing all variables "
         "with their values dictated by the dict assignment which maps "
         "Variable objects to float.");

  ///---------------------------------------------------------------------------
  ///                       \n Output Handling \n
  ///---------------------------------------------------------------------------

  qc.def(
      "qasm_str",
      [](qc::QuantumComputation& circ) {
        auto ss = std::stringstream();
        circ.dumpOpenQASM(ss);
        return ss.str();
      },
      "Get a OpenQASM 2.0 representation of the circuit. Note that this uses "
      "some custom extensions to OpenQASM 2.0 that allow for easier definition "
      "of multi-controlled gates. These extensions might not be supported by "
      "all OpenQASM 2.0 parsers.");
  qc.def(
      "qasm",
      [](qc::QuantumComputation& circ, const std::string& filename) {
        std::ofstream ofs(filename);
        circ.dumpOpenQASM(ofs);
        ofs.close();
      },
      "filename"_a,
      "Write a OpenQASM 2.0 representation of the circuit to the given file. "
      "Note that this uses some custom extensions to OpenQASM 2.0 that allow "
      "for easier definition of multi-controlled gates. These extensions "
      "might not be supported by all OpenQASM 2.0 parsers.");
  qc.def("__str__", [](qc::QuantumComputation& circ) {
    auto ss = std::stringstream();
    circ.print(ss);
    return ss.str();
  });
  qc.def("__repr__", [](qc::QuantumComputation& circ) {
    auto ss = std::stringstream();
    circ.print(ss);
    return ss.str();
  });

  ///---------------------------------------------------------------------------
  ///                            \n Operations \n
  ///---------------------------------------------------------------------------

  qc.def("gphase", &qc::QuantumComputation::gphase, "phase"_a,
         "Apply a global phase to the circuit.");

  qc.def("i", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::i), "q"_a,
         "Apply the identity on qubit q.");
  qc.def("i",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::i),
         "q"_a, "ctrl"_a,
         "Apply a controlled identity gate on qubit q with control ctrl.");
  qc.def("i",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::i),
         "q"_a, "controls"_a,
         "Apply a multi-controlled identity gate on qubit q with controls "
         "controls.");

  qc.def("h", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::h), "q"_a,
         "Apply the Hadamard gate on qubit q.");
  qc.def("h",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::h),
         "q"_a, "ctrl"_a,
         "Apply a controlled Hadamard gate on qubit q with control ctrl.");
  qc.def("h",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::h),
         "q"_a, "controls"_a,
         "Apply a multi-controlled Hadamard gate on qubit q with controls "
         "controls.");

  qc.def("x", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::x), "q"_a,
         "Apply an X gate on qubit q.");
  qc.def("x",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::x),
         "q"_a, "ctrl"_a,
         "Apply a controlled X gate on qubit q with control ctrl.");
  qc.def("x",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::x),
         "q"_a, "controls"_a,
         "Apply a multi-controlled X gate on qubit q with controls controls.");

  qc.def("y", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::y), "q"_a,
         "Apply a Y gate on qubit q.");
  qc.def("y",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::y),
         "q"_a, "ctrl"_a,
         "Apply a controlled Y gate on qubit q with control ctrl.");
  qc.def("y",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::y),
         "q"_a, "controls"_a,
         "Apply a multi-controlled Y gate on qubit q with controls controls.");

  qc.def("z", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::z), "q"_a,
         "Apply a Z gate on qubit q.");
  qc.def("z",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::z),
         "q"_a, "ctrl"_a,
         "Apply a controlled Z gate on qubit q with control ctrl.");
  qc.def("z",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::z),
         "q"_a, "controls"_a,
         "Apply a multi-controlled Z gate on qubit q with controls controls.");

  qc.def("s", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::s), "q"_a,
         "Apply an S gate on qubit q.");
  qc.def("s",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::s),
         "q"_a, "ctrl"_a,
         "Apply a controlled S gate on qubit q with control ctrl.");
  qc.def("s",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::s),
         "q"_a, "controls"_a,
         "Apply a multi-controlled S gate on qubit q with controls controls.");

  qc.def("sdag", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::sdag),
         "q"_a, "Apply an Sdag gate on qubit q.");
  qc.def("sdag",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::sdag),
         "q"_a, "ctrl"_a,
         "Apply a controlled Sdag gate on qubit q with control ctrl.");
  qc.def("sdag",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::sdag),
         "q"_a, "controls"_a,
         "Apply a multi-controlled Sdag gate on qubit q with controls "
         "controls.");

  qc.def("t", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::t), "q"_a,
         "Apply a T gate on qubit q.");
  qc.def("t",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::t),
         "q"_a, "ctrl"_a,
         "Apply a controlled T gate on qubit q with control ctrl.");
  qc.def("t",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::t),
         "q"_a, "controls"_a,
         "Apply a multi-controlled T gate on qubit q with controls controls.");

  qc.def("tdag", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::tdag),
         "q"_a, "Apply a Tdag gate on qubit q.");
  qc.def("tdag",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::tdag),
         "q"_a, "ctrl"_a,
         "Apply a controlled Tdag gate on qubit q with control ctrl.");
  qc.def("tdag",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::tdag),
         "q"_a, "controls"_a,
         "Apply a multi-controlled Tdag gate on qubit q with controls "
         "controls.");

  qc.def("v", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::v), "q"_a,
         "Apply a V gate on qubit q.");
  qc.def("v",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::v),
         "q"_a, "ctrl"_a,
         "Apply a controlled V gate on qubit q with control ctrl.");
  qc.def("v",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::v),
         "q"_a, "controls"_a,
         "Apply a multi-controlled V gate on qubit q with controls controls.");

  qc.def("vdag", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::vdag),
         "q"_a, "Apply a Vdag gate on qubit q.");
  qc.def("vdag",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::vdag),
         "q"_a, "ctrl"_a,
         "Apply a controlled Vdag gate on qubit q with control ctrl.");
  qc.def("vdag",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::vdag),
         "q"_a, "controls"_a,
         "Apply a multi-controlled Vdag gate on qubit q with controls "
         "controls.");

  qc.def("u3",
         py::overload_cast<qc::Qubit, const qc::fp, const qc::fp, const qc::fp>(
             &qc::QuantumComputation::u3),
         "q"_a, "theta"_a, "phi"_a, "lambda"_a,
         "Apply a U3 gate on qubit q with parameters theta, phi, lambda.");
  qc.def("u3",
         py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp,
                           const qc::fp, const qc::fp>(
             &qc::QuantumComputation::u3),
         "q"_a, "ctrl"_a, "theta"_a, "phi"_a, "lambda"_a,
         "Apply a controlled U3 gate on qubit q with control ctrl and "
         "parameters theta, phi, lambda.");
  qc.def("u3",
         py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp,
                           const qc::fp, const qc::fp>(
             &qc::QuantumComputation::u3),
         "q"_a, "controls"_a, "theta"_a, "phi"_a, "lambda"_a,
         "Apply a multi-controlled U3 gate on qubit q with controls controls "
         "and parameters theta, phi, lambda.");

  qc.def("u2",
         py::overload_cast<qc::Qubit, const qc::fp, const qc::fp>(
             &qc::QuantumComputation::u2),
         "q"_a, "phi"_a, "lambda"_a,
         "Apply a U2 gate on qubit q with parameters phi, lambda.");
  qc.def("u2",
         py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp,
                           const qc::fp>(&qc::QuantumComputation::u2),
         "q"_a, "ctrl"_a, "phi"_a, "lambda"_a,
         "Apply a controlled U2 gate on qubit q with control ctrl and "
         "parameters phi, lambda.");
  qc.def("u2",
         py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp,
                           const qc::fp>(&qc::QuantumComputation::u2),
         "q"_a, "controls"_a, "phi"_a, "lambda"_a,
         "Apply a multi-controlled U2 gate on qubit q with controls controls "
         "and parameters phi, lambda.");

  qc.def("phase",
         py::overload_cast<qc::Qubit, const qc::fp>(
             &qc::QuantumComputation::phase),
         "q"_a, "lambda"_a,
         "Apply a phase gate on qubit q with parameter lambda.");
  qc.def("phase",
         py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
             &qc::QuantumComputation::phase),
         "q"_a, "ctrl"_a, "lambda"_a,
         "Apply a controlled phase gate on qubit q with control ctrl and "
         "parameter lambda.");
  qc.def("phase",
         py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
             &qc::QuantumComputation::phase),
         "q"_a, "controls"_a, "lambda"_a,
         "Apply a multi-controlled phase gate on qubit q with controls "
         "controls and parameter lambda.");

  qc.def("sx", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::sx), "q"_a,
         "Apply a square root of X gate on qubit q.");
  qc.def("sx",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::sx),
         "q"_a, "ctrl"_a,
         "Apply a controlled square root of X gate on qubit q with control "
         "ctrl.");
  qc.def("sx",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::sx),
         "q"_a, "controls"_a,
         "Apply a multi-controlled square root of X gate on qubit q with "
         "controls controls.");

  qc.def("sxdag", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::sxdag),
         "q"_a, "Apply the inverse of the square root of X gate on qubit q.");
  qc.def("sxdag",
         py::overload_cast<qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::sxdag),
         "q"_a, "ctrl"_a,
         "Apply the controlled inverse of the square root of X gate on qubit "
         "q with control ctrl.");
  qc.def("sxdag",
         py::overload_cast<qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::sxdag),
         "q"_a, "controls"_a,
         "Apply the multi-controlled inverse of the square root of X gate on "
         "qubit q with controls controls.");

  qc.def(
      "rx",
      py::overload_cast<qc::Qubit, const qc::fp>(&qc::QuantumComputation::rx),
      "q"_a, "theta"_a,
      "Apply an X-rotation gate on qubit q with angle theta.");
  qc.def("rx",
         py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
             &qc::QuantumComputation::rx),
         "q"_a, "ctrl"_a, "theta"_a,
         "Apply a controlled X-rotation gate on qubit q with control ctrl "
         "and angle theta.");
  qc.def("rx",
         py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
             &qc::QuantumComputation::rx),
         "q"_a, "controls"_a, "theta"_a,
         "Apply a multi-controlled X-rotation gate on qubit q with controls "
         "controls and angle theta.");

  qc.def(
      "ry",
      py::overload_cast<qc::Qubit, const qc::fp>(&qc::QuantumComputation::ry),
      "q"_a, "theta"_a, "Apply a Y-rotation gate on qubit q with angle theta.");
  qc.def("ry",
         py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
             &qc::QuantumComputation::ry),
         "q"_a, "ctrl"_a, "theta"_a,
         "Apply a controlled Y-rotation gate on qubit q with control ctrl "
         "and angle theta.");
  qc.def("ry",
         py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
             &qc::QuantumComputation::ry),
         "q"_a, "controls"_a, "theta"_a,
         "Apply a multi-controlled Y-rotation gate on qubit q with controls "
         "controls and angle theta.");

  qc.def(
      "rz",
      py::overload_cast<qc::Qubit, const qc::fp>(&qc::QuantumComputation::rz),
      "q"_a, "phi"_a, "Apply a Z-rotation gate on qubit q with angle phi.");
  qc.def("rz",
         py::overload_cast<qc::Qubit, const qc::Control&, const qc::fp>(
             &qc::QuantumComputation::rz),
         "q"_a, "ctrl"_a, "phi"_a,
         "Apply a controlled Z-rotation gate on qubit q with control ctrl "
         "and angle phi.");
  qc.def("rz",
         py::overload_cast<qc::Qubit, const qc::Controls&, const qc::fp>(
             &qc::QuantumComputation::rz),
         "q"_a, "controls"_a, "phi"_a,
         "Apply a multi-controlled Z-rotation gate on qubit q with controls "
         "controls and angle phi.");

  qc.def("swap",
         py::overload_cast<qc::Qubit, qc::Qubit>(&qc::QuantumComputation::swap),
         "q1"_a, "q2"_a, "Apply a SWAP gate on qubits q1 and q2.");
  qc.def("swap",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::swap),
         "q1"_a, "q2"_a, "ctrl"_a,
         "Apply a controlled SWAP (Fredkin) gate on qubits q1 and q2 with "
         "control ctrl.");
  qc.def("swap",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::swap),
         "q1"_a, "q2"_a, "controls"_a,
         "Apply a multi-controlled SWAP gate on qubits q1 and q2 with "
         "controls controls.");

  qc.def(
      "iswap",
      py::overload_cast<qc::Qubit, qc::Qubit>(&qc::QuantumComputation::iswap),
      "q1"_a, "q2"_a, "Apply an iSWAP gate on qubits q1 and q2.");
  qc.def("iswap",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::iswap),
         "q1"_a, "q2"_a, "ctrl"_a,
         "Apply a controlled iSWAP gate on qubits q1 and q2 with control "
         "ctrl.");
  qc.def("iswap",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::iswap),
         "q1"_a, "q2"_a, "controls"_a,
         "Apply a multi-controlled iSWAP gate on qubits q1 and q2 with "
         "controls controls.");

  qc.def(
      "peres",
      py::overload_cast<qc::Qubit, qc::Qubit>(&qc::QuantumComputation::peres),
      "q1"_a, "q2"_a, "Apply a Peres gate on qubits q1 and q2.");
  qc.def("peres",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::peres),
         "q1"_a, "q2"_a, "ctrl"_a,
         "Apply a controlled Peres gate on qubits q1 and q2 with control "
         "ctrl.");
  qc.def("peres",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::peres),
         "q1"_a, "q2"_a, "controls"_a,
         "Apply a multi-controlled Peres gate on qubits q1 and q2 with "
         "controls controls.");

  qc.def("peresdag",
         py::overload_cast<qc::Qubit, qc::Qubit>(
             &qc::QuantumComputation::peresdag),
         "q1"_a, "q2"_a, "Apply an inverse Peres gate on qubits q1 and q2.");
  qc.def("peresdag",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::peresdag),
         "q1"_a, "q2"_a, "ctrl"_a,
         "Apply a controlled inverse Peres gate on qubits q1 and q2 with "
         "control ctrl.");
  qc.def("peresdag",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::peresdag),
         "q1"_a, "q2"_a, "controls"_a,
         "Apply a multi-controlled inverse Peres gate on qubits q1 and q2 "
         "with controls controls.");

  qc.def("dcx",
         py::overload_cast<qc::Qubit, qc::Qubit>(&qc::QuantumComputation::dcx),
         "q1"_a, "q2"_a, "Apply a double CNOT gate on qubits q1 and q2.");
  qc.def("dcx",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::dcx),
         "q1"_a, "q2"_a, "ctrl"_a,
         "Apply a controlled double CNOT gate on qubits q1 and q2 with "
         "control ctrl.");
  qc.def("dcx",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::dcx),
         "q1"_a, "q2"_a, "controls"_a,
         "Apply a multi-controlled double CNOT gate on qubits q1 and q2 with "
         "controls controls.");

  qc.def("ecr",
         py::overload_cast<qc::Qubit, qc::Qubit>(&qc::QuantumComputation::ecr),
         "q1"_a, "q2"_a,
         "Apply an echoed cross-resonance gate on qubits q1 and q2.");
  qc.def("ecr",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&>(
             &qc::QuantumComputation::ecr),
         "q1"_a, "q2"_a, "ctrl"_a,
         "Apply a controlled echoed cross-resonance gate on qubits q1 and q2 "
         "with control ctrl.");
  qc.def("ecr",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&>(
             &qc::QuantumComputation::ecr),
         "q1"_a, "q2"_a, "controls"_a,
         "Apply a multi-controlled echoed cross-resonance gate on qubits q1 "
         "and q2 with controls controls.");

  qc.def("rxx",
         py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
             &qc::QuantumComputation::rxx),
         "q1"_a, "q2"_a, "phi"_a,
         "Apply an XX-rotation gate on qubits q1 and q2 with angle phi.");
  qc.def("rxx",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
             &qc::QuantumComputation::rxx),
         "q1"_a, "q2"_a, "ctrl"_a, "phi"_a,
         "Apply a controlled XX-rotation gate on qubits q1 and q2 with "
         "control ctrl and angle phi.");
  qc.def("rxx",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
             &qc::QuantumComputation::rxx),
         "q1"_a, "q2"_a, "controls"_a, "phi"_a,
         "Apply a multi-controlled XX-rotation gate on qubits q1 and q2 with "
         "controls controls and angle phi.");

  qc.def("ryy",
         py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
             &qc::QuantumComputation::ryy),
         "q1"_a, "q2"_a, "phi"_a,
         "Apply a YY-rotation gate on qubits q1 and q2 with angle phi.");
  qc.def("ryy",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
             &qc::QuantumComputation::ryy),
         "q1"_a, "q2"_a, "ctrl"_a, "phi"_a,
         "Apply a controlled YY-rotation gate on qubits q1 and q2 with "
         "control ctrl and angle phi.");
  qc.def("ryy",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
             &qc::QuantumComputation::ryy),
         "q1"_a, "q2"_a, "controls"_a, "phi"_a,
         "Apply a multi-controlled YY-rotation gate on qubits q1 and q2 with "
         "controls controls and angle phi.");

  qc.def("rzz",
         py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
             &qc::QuantumComputation::rzz),
         "q1"_a, "q2"_a, "phi"_a,
         "Apply a ZZ-rotation gate on qubits q1 and q2 with angle phi.");
  qc.def("rzz",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
             &qc::QuantumComputation::rzz),
         "q1"_a, "q2"_a, "ctrl"_a, "phi"_a,
         "Apply a controlled ZZ-rotation gate on qubits q1 and q2 with "
         "control ctrl and angle phi.");
  qc.def("rzz",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
             &qc::QuantumComputation::rzz),
         "q1"_a, "q2"_a, "controls"_a, "phi"_a,
         "Apply a multi-controlled ZZ-rotation gate on qubits q1 and q2 with "
         "controls controls and angle phi.");

  qc.def("rzx",
         py::overload_cast<qc::Qubit, qc::Qubit, qc::fp>(
             &qc::QuantumComputation::rzx),
         "q1"_a, "q2"_a, "phi"_a,
         "Apply a ZX-rotation gate on qubits q1 and q2 with angle phi.");
  qc.def("rzx",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp>(
             &qc::QuantumComputation::rzx),
         "q1"_a, "q2"_a, "ctrl"_a, "phi"_a,
         "Apply a controlled ZX-rotation gate on qubits q1 and q2 with "
         "control ctrl and angle phi.");
  qc.def("rzx",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp>(
             &qc::QuantumComputation::rzx),
         "q1"_a, "q2"_a, "controls"_a, "phi"_a,
         "Apply a multi-controlled ZX-rotation gate on qubits q1 and q2 with "
         "controls controls and angle phi.");

  qc.def("xx_minus_yy",
         py::overload_cast<qc::Qubit, qc::Qubit, qc::fp, qc::fp>(
             &qc::QuantumComputation::xx_minus_yy),
         "q1"_a, "q2"_a, "phi"_a, "lambda"_a,
         "Apply an XX-YY-rotation gate on qubits q1 and q2 with angles phi "
         "and lambda.");
  qc.def("xx_minus_yy",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp,
                           qc::fp>(&qc::QuantumComputation::xx_minus_yy),
         "q1"_a, "q2"_a, "ctrl"_a, "phi"_a, "lambda"_a,
         "Apply a controlled XX-YY-rotation gate on qubits q1 and q2 with "
         "control ctrl and angles phi and lambda.");
  qc.def("xx_minus_yy",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp,
                           qc::fp>(&qc::QuantumComputation::xx_minus_yy),
         "q1"_a, "q2"_a, "controls"_a, "phi"_a, "lambda"_a,
         "Apply a multi-controlled XX-YY-rotation gate on qubits q1 and q2 "
         "with controls controls and angles phi and lambda.");

  qc.def("xx_plus_yy",
         py::overload_cast<qc::Qubit, qc::Qubit, qc::fp, qc::fp>(
             &qc::QuantumComputation::xx_plus_yy),
         "q1"_a, "q2"_a, "phi"_a, "lambda"_a,
         "Apply an XX+YY-rotation gate on qubits q1 and q2 with angles phi "
         "and lambda.");
  qc.def("xx_plus_yy",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Control&, qc::fp,
                           qc::fp>(&qc::QuantumComputation::xx_plus_yy),
         "q1"_a, "q2"_a, "ctrl"_a, "phi"_a, "lambda"_a,
         "Apply a controlled XX+YY-rotation gate on qubits q1 and q2 with "
         "control ctrl and angles phi and lambda.");
  qc.def("xx_plus_yy",
         py::overload_cast<qc::Qubit, qc::Qubit, const qc::Controls&, qc::fp,
                           qc::fp>(&qc::QuantumComputation::xx_plus_yy),
         "q1"_a, "q2"_a, "controls"_a, "phi"_a, "lambda"_a,
         "Apply a multi-controlled XX+YY-rotation gate on qubits q1 and q2 "
         "with controls controls and angles phi and lambda.");

  qc.def("measure",
         py::overload_cast<qc::Qubit, std::size_t>(
             &qc::QuantumComputation::measure),
         "q"_a, "c"_a,
         "Measure qubit q and store the result in classical register c.");
  qc.def("measure",
         py::overload_cast<qc::Qubit, const std::pair<std::string, qc::Bit>&>(
             &qc::QuantumComputation::measure),
         "q"_a, "c"_a,
         "Measure qubit q and store the result in a named classical register "
         "c.");
  qc.def("measure",
         py::overload_cast<const std::vector<qc::Qubit>&,
                           const std::vector<qc::Bit>&>(
             &qc::QuantumComputation::measure),
         "qs"_a, "cs"_a,
         "Measure qubits qs and store the result in classical register cs.");

  qc.def("reset", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::reset),
         "q"_a, "Reset qubit q.");
  qc.def("reset",
         py::overload_cast<const std::vector<qc::Qubit>&>(
             &qc::QuantumComputation::reset),
         "qs"_a, "Reset qubits qs.");

  qc.def("barrier",
         py::overload_cast<qc::Qubit>(&qc::QuantumComputation::barrier), "q"_a,
         "Apply a barrier on qubit q.");
  qc.def("barrier",
         py::overload_cast<const std::vector<qc::Qubit>&>(
             &qc::QuantumComputation::barrier),
         "qs"_a, "Apply a barrier on qubits qs.");

  qc.def("classic_controlled",
         py::overload_cast<const qc::OpType, const qc::Qubit,
                           const qc::ClassicalRegister&, const std::uint64_t,
                           const std::vector<qc::fp>&>(
             &qc::QuantumComputation::classicControlled),
         "op"_a, "target"_a, "classical_register"_a, "expected_value"_a,
         "params"_a,
         "Apply a single-qubit operation if the classical register has the "
         "expected value.");
  qc.def("classic_controlled",
         py::overload_cast<const qc::OpType, const qc::Qubit, const qc::Control,
                           const qc::ClassicalRegister&, const std::uint64_t,
                           const std::vector<qc::fp>&>(
             &qc::QuantumComputation::classicControlled),
         "op"_a, "target"_a, "control"_a, "classical_register"_a,
         "expected_value"_a, "params"_a,
         "Apply a controlled single-qubit operation if the classical "
         "register has the expected value.");
  qc.def("classic_controlled",
         py::overload_cast<const qc::OpType, const qc::Qubit,
                           const qc::Controls&, const qc::ClassicalRegister&,
                           const std::uint64_t, const std::vector<qc::fp>&>(
             &qc::QuantumComputation::classicControlled),
         "op"_a, "target"_a, "controls"_a, "classical_register"_a,
         "expected_value"_a, "params"_a,
         "Apply a multi-controlled single-qubit operation if the classical "
         "register has the expected value.");
}
} // namespace mqt
