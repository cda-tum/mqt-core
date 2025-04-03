/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/operations/Control.hpp"
#include "python/pybind11.hpp"

#include <pybind11/operators.h>

namespace mqt {

void registerControl(const py::module& m) {

  auto control = py::class_<qc::Control>(m, "Control");
  auto controlType = py::enum_<qc::Control::Type>(control, "Type");
  controlType.value("Pos", qc::Control::Type::Pos);
  controlType.value("Neg", qc::Control::Type::Neg);
  controlType.def(
      "__str__",
      [](const qc::Control::Type& type) {
        return type == qc::Control::Type::Pos ? "Pos" : "Neg";
      },
      py::prepend());
  controlType.def(
      "__repr__",
      [](const qc::Control::Type& type) {
        return type == qc::Control::Type::Pos ? "Pos" : "Neg";
      },
      py::prepend());
  controlType.def("__bool__", [](const qc::Control::Type& type) {
    return type == qc::Control::Type::Pos;
  });
  py::implicitly_convertible<py::bool_, qc::Control::Type>();
  py::implicitly_convertible<py::str, qc::Control::Type>();

  control.def(py::init<qc::Qubit, qc::Control::Type>(), "qubit"_a,
              "type_"_a = qc::Control::Type::Pos);
  control.def_readwrite("type_", &qc::Control::type);
  control.def_readwrite("qubit", &qc::Control::qubit);
  control.def("__str__", [](const qc::Control& c) { return c.toString(); });
  control.def("__repr__", [](const qc::Control& c) { return c.toString(); });
  control.def(py::self == py::self);
  control.def(py::self != py::self);
  control.def(hash(py::self));
  py::implicitly_convertible<py::int_, qc::Control>();
}

} // namespace mqt
