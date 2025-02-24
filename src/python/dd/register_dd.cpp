/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDpackageConfig.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "ir/QuantumComputation.hpp"
#include "python/pybind11.hpp"

#include <cstddef>
#include <pybind11/numpy.h>

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

// forward declarations
void registerVectorDDs(const py::module& mod);
void registerMatrixDDs(const py::module& mod);
void registerDDPackage(const py::module& mod);

PYBIND11_MODULE(dd, mod, py::mod_gil_not_used()) {
  // Vector Decision Diagrams
  registerVectorDDs(mod);

  // Matrix Decision Diagrams
  registerMatrixDDs(mod);

  // DD Package
  registerDDPackage(mod);

  mod.def(
      "sample",
      [](const qc::QuantumComputation& qc, const size_t shots = 1024U,
         const size_t seed = 0U) { return dd::sample(qc, shots, seed); },
      "qc"_a, "shots"_a = 1024U, "seed"_a = 0U);

  mod.def("simulate", &dd::simulate<dd::DDPackageConfig>, "qc"_a,
          "initial_state"_a, "dd_package"_a);

  mod.def(
      "build_functionality",
      [](const qc::QuantumComputation& qc, dd::Package<>& p,
         const bool recursive = false) {
        if (recursive) {
          return dd::buildFunctionalityRecursive(qc, p);
        }
        return dd::buildFunctionality(qc, p);
      },
      "qc"_a, "dd_package"_a, "recursive"_a = false);
}

} // namespace mqt
