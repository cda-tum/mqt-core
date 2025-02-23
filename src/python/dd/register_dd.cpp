/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Definitions.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "python/pybind11.hpp"

#include <cstddef>
#include <memory>
#include <pybind11/numpy.h>

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

// forward declarations
void registerVectorDDs(const py::module& mod);
void registerMatrixDDs(const py::module& mod);

PYBIND11_MODULE(dd, mod, py::mod_gil_not_used()) {
  // Vector Decision Diagrams
  registerVectorDDs(mod);

  // Matrix Decision Diagrams
  registerMatrixDDs(mod);

  // DD Package
  auto dd = py::class_<dd::Package<>, std::unique_ptr<dd::Package<>>>(
      mod, "DDPackage");

  // Constructor
  dd.def(py::init<size_t>(), "num_qubits"_a = dd::Package<>::DEFAULT_QUBITS);

  // Resizing the package
  dd.def("resize", &dd::Package<>::resize, "num_qubits"_a);

  // Getting the number of qubits the package is configured for
  dd.def("max_qubits", &dd::Package<>::qubits);

  // Triggering garbage collection
  dd.def("garbage_collect", &dd::Package<>::garbageCollect, "force"_a = false);
}

} // namespace mqt
