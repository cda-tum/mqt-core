#include "operations/NonUnitaryOperation.hpp"
#include "python/pybind11.hpp"

namespace mqt {

void registerNonUnitaryOperation(py::module& m) {
  py::class_<qc::NonUnitaryOperation, qc::Operation>(
      m, "NonUnitaryOperation",
      "Non-unitary operations such as measurements and resets.")
      .def(
          py::init<std::size_t, std::vector<qc::Qubit>, std::vector<qc::Bit>>(),
          "nq"_a, "targets"_a, "classics"_a,
          "Create a multi-qubit measurement operation.")
      .def(py::init<std::size_t, qc::Qubit, qc::Bit>(), "nq"_a, "target"_a,
           "classic"_a,
           "Create a measurement operation that measures `target` into "
           "`classic`.")
      .def(py::init<std::size_t, std::vector<qc::Qubit>, qc::OpType>(), "nq"_a,
           "targets"_a, "op_type"_a = qc::OpType::Reset,
           "Create a multi-qubit reset operation.")
      .def_property_readonly(
          "classics",
          py::overload_cast<>(&qc::NonUnitaryOperation::getClassics,
                              py::const_),
          "Return the classical bits.");
}

} // namespace mqt
