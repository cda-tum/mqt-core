/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Definitions.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/Operation.hpp"
#include "python/pybind11.hpp"

#include <cstdint>
#include <memory>
#include <sstream>

namespace mqt {

void registerClassicControlledOperation(py::module& m) {
  py::enum_<qc::ComparisonKind>(m, "ComparisonKind")
      .value("eq", qc::ComparisonKind::Eq)
      .value("neq", qc::ComparisonKind::Neq)
      .value("lt", qc::ComparisonKind::Lt)
      .value("leq", qc::ComparisonKind::Leq)
      .value("gt", qc::ComparisonKind::Gt)
      .value("geq", qc::ComparisonKind::Geq)
      .export_values()
      .def("__str__",
           [](const qc::ComparisonKind& cmp) { return qc::toString(cmp); })
      .def("__repr__",
           [](const qc::ComparisonKind& cmp) { return qc::toString(cmp); });

  auto ccop = py::class_<qc::ClassicControlledOperation, qc::Operation>(
      m, "ClassicControlledOperation");

  ccop.def(
      py::init([](qc::Operation* operation, qc::ClassicalRegister controlReg,
                  std::uint64_t expectedVal, qc::ComparisonKind cmp) {
        return std::make_unique<qc::ClassicControlledOperation>(
            operation->clone(), controlReg, expectedVal, cmp);
      }),
      "operation"_a, "control_register"_a, "expected_value"_a = 1U,
      "comparison_kind"_a = qc::ComparisonKind::Eq);
  ccop.def_property_readonly("operation",
                             &qc::ClassicControlledOperation::getOperation,
                             py::return_value_policy::reference_internal);
  ccop.def_property_readonly(
      "control_register", &qc::ClassicControlledOperation::getControlRegister);
  ccop.def_property_readonly("expected_value",
                             &qc::ClassicControlledOperation::getExpectedValue);
  ccop.def_property_readonly(
      "comparison_kind", &qc::ClassicControlledOperation::getComparisonKind);
  ccop.def("__repr__", [](const qc::ClassicControlledOperation& op) {
    std::stringstream ss;
    const auto& controlReg = op.getControlRegister();
    ss << "ClassicControlledOperation(<...op...>, "
       << "control_register=(" << controlReg.first << ", " << controlReg.second
       << "), "
       << "expected_value=" << op.getExpectedValue() << ", "
       << "comparison_kind='" << op.getComparisonKind() << "')";
    return ss.str();
  });
}

} // namespace mqt
