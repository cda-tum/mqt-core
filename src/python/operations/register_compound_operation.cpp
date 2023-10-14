#include "operations/CompoundOperation.hpp"
#include "python/pybind11.hpp"

namespace mqt {

using DiffType = std::vector<std::unique_ptr<qc::Operation>>::difference_type;
using SizeType = std::vector<std::unique_ptr<qc::Operation>>::size_type;

void registerCompoundOperation(py::module& m) {
  auto wrap = [](DiffType i, const SizeType size) {
    if (i < 0) {
      i += static_cast<DiffType>(size);
    }
    if (i < 0 || static_cast<SizeType>(i) >= size) {
      throw py::index_error();
    }
    return i;
  };

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
          [&wrap](const qc::CompoundOperation& op, DiffType i) {
            i = wrap(i, op.size());
            return op.at(static_cast<SizeType>(i)).get();
          },
          py::return_value_policy::reference_internal, "i"_a,
          "Return i-th sub-operation. Beware: this gives write access to the "
          "sub-operation.")
      .def(
          "__getitem__",
          [](qc::CompoundOperation& op, const py::slice& slice) {
            std::size_t start{};
            std::size_t stop{};
            std::size_t step{};
            std::size_t sliceLength{};
            if (!slice.compute(op.size(), &start, &stop, &step, &sliceLength)) {
              throw py::error_already_set();
            }
            auto ops = std::vector<qc::Operation*>();
            ops.reserve(sliceLength);
            for (std::size_t i = start; i < stop; i += step) {
              ops.emplace_back(op.at(i).get());
            }
            return ops;
          },
          py::return_value_policy::reference_internal, "slice"_a,
          "Return a slice of the compound operation. Beware: this gives write "
          "access to the sub-operations.")
      .def(
          "append",
          [](qc::CompoundOperation& compOp, const qc::Operation& op) {
            compOp.emplace_back(op.clone());
          },
          "op"_a, "Append operation op to the `CompoundOperation`.")
      .def("empty", &qc::CompoundOperation::empty)
      .def("__repr__", [](const qc::CompoundOperation& op) {
        std::stringstream ss;
        ss << "CompoundOperation(" << op.getNqubits() << ", [...ops...])";
        return ss.str();
      });
}
} // namespace mqt
