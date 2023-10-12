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
  qc.def(py::init<std::size_t, std::size_t>(), "nq"_a, "nc"_a = 0U,
         "Constructs an empty QuantumComputation with the given number of "
         "qubits and classical bits.");
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

#define DEFINE_SINGLE_TARGET_OPERATION(op)                                     \
  qc.def(#op, &qc::QuantumComputation::op, "q"_a, "Add a " #op "(q) gate.");   \
  qc.def("c" #op, &qc::QuantumComputation::c##op, "control"_a, "target"_a,     \
         "Add a controlled-" #op "(control, target) gate.");                   \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, "controls"_a, "target"_a,  \
         "Add a multi-controlled-" #op " (controls, target) gate.");

  DEFINE_SINGLE_TARGET_OPERATION(i)
  DEFINE_SINGLE_TARGET_OPERATION(x)
  DEFINE_SINGLE_TARGET_OPERATION(y)
  DEFINE_SINGLE_TARGET_OPERATION(z)
  DEFINE_SINGLE_TARGET_OPERATION(h)
  DEFINE_SINGLE_TARGET_OPERATION(s)
  DEFINE_SINGLE_TARGET_OPERATION(sdg)
  DEFINE_SINGLE_TARGET_OPERATION(t)
  DEFINE_SINGLE_TARGET_OPERATION(tdg)
  DEFINE_SINGLE_TARGET_OPERATION(v)
  DEFINE_SINGLE_TARGET_OPERATION(vdg)
  DEFINE_SINGLE_TARGET_OPERATION(sx)
  DEFINE_SINGLE_TARGET_OPERATION(sxdg)

#define DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(op, param)             \
  qc.def(#op, &qc::QuantumComputation::op, py::arg(#param), "q"_a,             \
         "Add a `" #op "(" #param ", q)` gate.");                              \
  qc.def("c" #op, &qc::QuantumComputation::c##op, py::arg(#param),             \
         "control"_a, "target"_a,                                              \
         "Add a `controlled-" #op "(" #param ", control, target)` gate.");     \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, py::arg(#param),           \
         "controls"_a, "target"_a,                                             \
         "Add a `multi-controlled-" #op "(" #param                             \
         ", controls, target)` gate.");

  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rx, theta)
  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(ry, theta)
  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rz, theta)
  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(p, theta)

#define DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)       \
  qc.def(#op, &qc::QuantumComputation::op, py::arg(#param0), py::arg(#param1), \
         "q"_a, "Add a `" #op "(" #param0 ", " #param1 ", q)` gate.");         \
  qc.def("c" #op, &qc::QuantumComputation::c##op, py::arg(#param0),            \
         py::arg(#param1), "control"_a, "target"_a,                            \
         "Add a `controlled-" #op "(" #param0 ", " #param1                     \
         ", control, target)` gate.");                                         \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, py::arg(#param0),          \
         py::arg(#param1), "controls"_a, "target"_a,                           \
         "Add a `multi-controlled-" #op "(" #param0 ", " #param1               \
         ", controls, target)` gate.");

  DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(u2, phi, lambda_)

#define DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(op, param0, param1,     \
                                                       param2)                 \
  qc.def(#op, &qc::QuantumComputation::op, py::arg(#param0), py::arg(#param1), \
         py::arg(#param2), "q"_a,                                              \
         "Add a `" #op "(" #param0 ", " #param1 ", " #param2 ", q)` gate.");   \
  qc.def("c" #op, &qc::QuantumComputation::c##op, py::arg(#param0),            \
         py::arg(#param1), py::arg(#param2), "control"_a, "target"_a,          \
         "Add a `controlled-" #op "(" #param0 ", " #param1 ", " #param2        \
         ", control, target)` gate.");                                         \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, py::arg(#param0),          \
         py::arg(#param1), py::arg(#param2), "controls"_a, "target"_a,         \
         "Add a `multi-controlled-" #op "(" #param0 ", " #param1 ", " #param2  \
         ", controls, target)` gate.");

  DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(u, theta, phi, lambda_)

#define DEFINE_TWO_TARGET_OPERATION(op)                                        \
  qc.def(#op, &qc::QuantumComputation::op, "target1"_a, "target2"_a,           \
         "Add a `" #op "(target1, target2)` gate.");                           \
  qc.def("c" #op, &qc::QuantumComputation::c##op, "control"_a, "target1"_a,    \
         "target2"_a,                                                          \
         "Add a `controlled-" #op "(control, target1, target2)` gate.");       \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, "controls"_a, "target1"_a, \
         "target2"_a,                                                          \
         "Add a `multi-controlled-" #op                                        \
         " (controls, target1, target2)` gate.");

  DEFINE_TWO_TARGET_OPERATION(swap)
  DEFINE_TWO_TARGET_OPERATION(dcx)
  DEFINE_TWO_TARGET_OPERATION(ecr)
  DEFINE_TWO_TARGET_OPERATION(iswap)
  DEFINE_TWO_TARGET_OPERATION(peres)
  DEFINE_TWO_TARGET_OPERATION(peresdg)

#define DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(op, param)                \
  qc.def(#op, &qc::QuantumComputation::op, py::arg(#param), "target1"_a,       \
         "target2"_a, "Add a `" #op "(" #param ", target1, target2)` gate.");  \
  qc.def("c" #op, &qc::QuantumComputation::c##op, py::arg(#param),             \
         "control"_a, "target1"_a, "target2"_a,                                \
         "Add a `controlled-" #op "(" #param                                   \
         ", control, target1, target2)` gate.");                               \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, py::arg(#param),           \
         "controls"_a, "target1"_a, "target2"_a,                               \
         "Add a `multi-controlled-" #op "(" #param                             \
         ", controls, target1, target2)` gate.");

  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rxx, theta)
  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(ryy, theta)
  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzz, theta)
  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzx, theta)

#define DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)          \
  qc.def(#op, &qc::QuantumComputation::op, py::arg(#param0), py::arg(#param1), \
         "target1"_a, "target2"_a,                                             \
         "Add a `" #op "(" #param0 ", " #param1 ", target1, target2)` "        \
         "gate.");                                                             \
  qc.def("c" #op, &qc::QuantumComputation::c##op, py::arg(#param0),            \
         py::arg(#param1), "control"_a, "target1"_a, "target2"_a,              \
         "Add a `controlled-" #op "(" #param0 ", " #param1                     \
         ", control, target1, target2)` gate.");                               \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, py::arg(#param0),          \
         py::arg(#param1), "controls"_a, "target1"_a, "target2"_a,             \
         "Add a `multi-controlled-" #op "(" #param0 ", " #param1               \
         ", controls, target1, target2)` gate.");

  DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_minus_yy, theta, beta)
  DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_plus_yy, theta, beta)

#undef DEFINE_SINGLE_TARGET_OPERATION
#undef DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION
#undef DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION
#undef DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION
#undef DEFINE_TWO_TARGET_OPERATION
#undef DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION
#undef DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION

  qc.def("gphase", &qc::QuantumComputation::gphase, "phase"_a,
         "Add a global `phase` to the circuit.");

  qc.def("measure",
         py::overload_cast<qc::Qubit, std::size_t>(
             &qc::QuantumComputation::measure),
         "qubit"_a, "cbit"_a, "Add a `measure(qubit, cbit)` gate.");
  qc.def("measure",
         py::overload_cast<qc::Qubit, const std::pair<std::string, qc::Bit>&>(
             &qc::QuantumComputation::measure),
         "qubit"_a, "creg_bit"_a, "Add a `measure(qubit, creg[bit])` gate.");
  qc.def("measure",
         py::overload_cast<const std::vector<qc::Qubit>&,
                           const std::vector<qc::Bit>&>(
             &qc::QuantumComputation::measure),
         "qubits"_a, "cbits"_a,
         "Add a `measure(qubits, cbits)` gate that measures all qubits in "
         "`qubits` and stores the result in the classical bits in `cbits`.");
  qc.def("measure_all", &qc::QuantumComputation::measureAll,
         "add_bits"_a = true,
         "Add measurements to all qubits. If `add_bits` is true, add a new "
         "classical register (named \"meas\") with the same size as the number "
         "of qubits and store the measurement results in there. Otherwise, "
         "store the measurement results in the existing classical bits.");

  qc.def("reset", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::reset),
         "q"_a, "Add a `reset(q)` gate.");
  qc.def("reset",
         py::overload_cast<const std::vector<qc::Qubit>&>(
             &qc::QuantumComputation::reset),
         "qubits"_a,
         "Add `reset(qs)` gate that resets all qubits in `qubits`.");

  qc.def("barrier", py::overload_cast<>(&qc::QuantumComputation::barrier),
         "Add a `barrier()` gate.");
  qc.def("barrier",
         py::overload_cast<qc::Qubit>(&qc::QuantumComputation::barrier), "q"_a,
         "Add a `barrier(q)` gate.");
  qc.def("barrier",
         py::overload_cast<const std::vector<qc::Qubit>&>(
             &qc::QuantumComputation::barrier),
         "qubits"_a,
         "Add a `barrier(qs)` gate that acts as a barrier for all qubits "
         "in `qubits`.");

  qc.def("classic_controlled",
         py::overload_cast<const qc::OpType, const qc::Qubit,
                           const qc::ClassicalRegister&, const std::uint64_t,
                           const std::vector<qc::fp>&>(
             &qc::QuantumComputation::classicControlled),
         "op"_a, "target"_a, "creg"_a, "expected_value"_a = 1U,
         "params"_a = std::vector<qc::fp>{},
         "Add a `op(params, target).c_if(creg, expected_value)` gate.");
  qc.def(
      "classic_controlled",
      py::overload_cast<const qc::OpType, const qc::Qubit, const qc::Control,
                        const qc::ClassicalRegister&, const std::uint64_t,
                        const std::vector<qc::fp>&>(
          &qc::QuantumComputation::classicControlled),
      "op"_a, "target"_a, "control"_a, "creg"_a, "expected_value"_a = 1U,
      "params"_a = std::vector<qc::fp>{},
      "Add a `cop(params, control, target).c_if(creg, expected_value)` gate.");
  qc.def("classic_controlled",
         py::overload_cast<const qc::OpType, const qc::Qubit,
                           const qc::Controls&, const qc::ClassicalRegister&,
                           const std::uint64_t, const std::vector<qc::fp>&>(
             &qc::QuantumComputation::classicControlled),
         "op"_a, "target"_a, "controls"_a, "creg"_a, "expected_value"_a = 1U,
         "params"_a = std::vector<qc::fp>{},
         "Add a `mcop(params, controls, target).c_if(creg, expected_value)` "
         "gate.");
}
} // namespace mqt
