#include "operations/SymbolicOperation.hpp"
#include "python/pybind11.hpp"

namespace mqt {

void registerSymbolicOperation(py::module& m) {
  py::class_<qc::SymbolicOperation, qc::StandardOperation>(
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
      .def("get_instantiated_operation",
           &qc::SymbolicOperation::getInstantiatedOperation,
           "assignment"_a
           "Return a `StandardOperation` version of this operation that is "
           "obtained by replacing all variables by their values dictated by the"
           " dict assignment which maps Variable objects to float.")
      .def("instantiate", &qc::SymbolicOperation::instantiate,
           "assignment"_a
           "Replace all variables within this operation by their values "
           "dictated by the dict assignment which maps Variable objects to "
           "float.");
}

} // namespace mqt
