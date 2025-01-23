/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Operation.hpp"
#include "python/pybind11.hpp"

#include <cstddef>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace mqt {

using DiffType = std::vector<std::unique_ptr<qc::Operation>>::difference_type;
using SizeType = std::vector<std::unique_ptr<qc::Operation>>::size_type;

void registerCompoundOperation(const py::module& m) {
  auto wrap = [](DiffType i, const SizeType size) {
    if (i < 0) {
      i += static_cast<DiffType>(size);
    }
    if (i < 0 || static_cast<SizeType>(i) >= size) {
      throw py::index_error();
    }
    return i;
  };

  py::class_<qc::CompoundOperation, qc::Operation>(m, "CompoundOperation")
      .def(py::init<>())
      .def(py::init([](const std::vector<qc::Operation*>& ops) {
             std::vector<std::unique_ptr<qc::Operation>> uniqueOps;
             uniqueOps.reserve(ops.size());
             for (auto& op : ops) {
               uniqueOps.emplace_back(op->clone());
             }
             return qc::CompoundOperation(std::move(uniqueOps));
           }),
           "ops"_a)
      .def("__len__", &qc::CompoundOperation::size)
      .def(
          "__getitem__",
          [&wrap](const qc::CompoundOperation& op, DiffType i) {
            i = wrap(i, op.size());
            return op.at(static_cast<SizeType>(i)).get();
          },
          py::return_value_policy::reference_internal, "i"_a)
      .def(
          "__getitem__",
          [](const qc::CompoundOperation& op, const py::slice& slice) {
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
          py::return_value_policy::reference_internal, "slice"_a)
      .def(
          "append",
          [](qc::CompoundOperation& compOp, const qc::Operation& op) {
            compOp.emplace_back(op.clone());
          },
          "op"_a)
      .def("empty", &qc::CompoundOperation::empty)
      .def("__repr__", [](const qc::CompoundOperation& op) {
        std::stringstream ss;
        ss << "CompoundOperation([..." << op.size() << " ops...])";
        return ss.str();
      });
}
} // namespace mqt
