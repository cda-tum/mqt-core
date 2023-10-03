#include "operations/CompoundOperation.hpp"
#include "python/pybind11.hpp"

namespace mqt {

void registerCompoundOperation(py::module& m) {
  py::class_<qc::CompoundOperation, qc::Operation>(
      m, "CompoundOperation",
      "Quantum operation comprised of multiple sub-operations.")
      .def(py::init<std::size_t>(), "nq"_a,
           "Create an empty compound operation on `nq` qubits.")
      .def(py::init([](std::size_t nq, std::vector<qc::Operation*>& ops) {
             std::vector<std::unique_ptr<qc::Operation>> uniqueOps;
             uniqueOps.reserve(ops.size());
             for (auto& op : ops) {
               uniqueOps.emplace_back(op->clone());
             }
             return qc::CompoundOperation(nq, std::move(uniqueOps));
           }),
           "nq"_a, "ops"_a,
           "Create a compound operation from a list of operations.")
      .def("__len__", &qc::CompoundOperation::size,
           "Return number of sub-operations.")
      .def(
          "__getitem__",
          [](const qc::CompoundOperation& op, std::size_t i) {
            return op.at(i).get();
          },
          py::return_value_policy::reference_internal, "i"_a,
          "Return i-th sub-operation. Beware: this gives write access to the "
          "sub-operation.")
      .def(
          "append",
          [](qc::CompoundOperation& compOp, const qc::Operation& op) {
            compOp.emplace_back(op.clone());
          },
          "op"_a, "Append operation op to the `CompoundOperation`.")
      .def("empty", &qc::CompoundOperation::empty);
}
} // namespace mqt
