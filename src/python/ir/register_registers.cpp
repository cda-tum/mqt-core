/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Definitions.hpp"
#include "ir/Register.hpp"
#include "python/pybind11.hpp"

#include <cstddef>
#include <pybind11/operators.h>
#include <string>

namespace mqt {

void registerRegisters(py::module& m) {
  py::class_<qc::QuantumRegister>(m, "QuantumRegister")
      .def(py::init<const qc::Qubit, const std::size_t, const std::string&>(),
           "start"_a, "size"_a, "name"_a = "")
      .def_property_readonly("name", &qc::QuantumRegister::getName)
      .def_property(
          "start",
          [](const qc::QuantumRegister& reg) { return reg.getStartIndex(); },
          [](qc::QuantumRegister& reg, const qc::Qubit start) {
            reg.getStartIndex() = start;
          })
      .def_property(
          "size", [](const qc::QuantumRegister& reg) { return reg.getSize(); },
          [](qc::QuantumRegister& reg, const std::size_t size) {
            reg.getSize() = size;
          })
      .def_property_readonly("end", &qc::QuantumRegister::getEndIndex)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(hash(py::self))
      .def("__getitem__", &qc::QuantumRegister::getGlobalIndex, "key"_a)
      .def("__contains__", &qc::QuantumRegister::contains)
      .def("__repr__", [](const qc::QuantumRegister& reg) {
        return "QuantumRegister(name=" + reg.getName() +
               ", start=" + std::to_string(reg.getStartIndex()) +
               ", size=" + std::to_string(reg.getSize()) + ")";
      });

  py::class_<qc::ClassicalRegister>(m, "ClassicalRegister")
      .def(py::init<const qc::Bit, const std::size_t, const std::string&>(),
           "start"_a, "size"_a, "name"_a = "")
      .def_property_readonly("name", &qc::ClassicalRegister::getName)
      .def_property(
          "start",
          [](const qc::ClassicalRegister& reg) { return reg.getStartIndex(); },
          [](qc::ClassicalRegister& reg, const qc::Bit start) {
            reg.getStartIndex() = start;
          })
      .def_property(
          "size",
          [](const qc::ClassicalRegister& reg) { return reg.getSize(); },
          [](qc::ClassicalRegister& reg, const std::size_t size) {
            reg.getSize() = size;
          })
      .def_property_readonly("end", &qc::ClassicalRegister::getEndIndex)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(hash(py::self))
      .def("__getitem__", &qc::ClassicalRegister::getGlobalIndex, "key"_a)
      .def("__contains__", &qc::ClassicalRegister::contains)
      .def("__repr__", [](const qc::ClassicalRegister& reg) {
        return "ClassicalRegister(name=" + reg.getName() +
               ", start=" + std::to_string(reg.getStartIndex()) +
               ", size=" + std::to_string(reg.getSize()) + ")";
      });
}

} // namespace mqt
