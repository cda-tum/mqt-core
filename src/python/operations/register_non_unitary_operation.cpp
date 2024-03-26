#include "operations/NonUnitaryOperation.hpp"
#include "python/pybind11.hpp"

namespace mqt {

void registerNonUnitaryOperation(py::module& m) {
  py::class_<qc::NonUnitaryOperation, qc::Operation>(
      m, "NonUnitaryOperation",
      "Non-unitary operations such as measurements and resets.")
      .def(py::init<std::vector<qc::Qubit>, std::vector<qc::Bit>>(),
           "targets"_a, "classics"_a,
           "Create a multi-qubit measurement operation.")
      .def(py::init<qc::Qubit, qc::Bit>(), "target"_a, "classic"_a,
           "Create a measurement operation that measures `target` into "
           "`classic`.")
      .def(py::init<std::vector<qc::Qubit>, qc::OpType>(), "targets"_a,
           "op_type"_a = qc::OpType::Reset,
           "Create a multi-qubit reset operation.")
      .def_property_readonly(
          "classics",
          py::overload_cast<>(&qc::NonUnitaryOperation::getClassics,
                              py::const_),
          "Return the classical bits.")
      .def("__repr__", [](const qc::NonUnitaryOperation& op) {
        std::stringstream ss;
        ss << "NonUnitaryOperation(";
        const auto& targets = op.getTargets();
        if (targets.size() == 1U) {
          ss << "target=" << targets[0];
        } else {
          ss << "targets=[";
          for (const auto& target : targets) {
            ss << target << ", ";
          }
          ss << "]";
        }
        const auto& classics = op.getClassics();
        if (!classics.empty()) {
          ss << ", ";
          if (classics.size() == 1U) {
            ss << "classic=" << classics[0];
          } else {
            ss << "classics=[";
            for (const auto& classic : classics) {
              ss << classic << ", ";
            }
            ss << "]";
          }
        }
        ss << ")";
        return ss.str();
      });
}

} // namespace mqt
