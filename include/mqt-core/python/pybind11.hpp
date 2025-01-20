/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// This file must be the first include for any bindings code.

#pragma once

#include <pybind11/pybind11.h> // IWYU pragma: export
#include <pybind11/stl.h>      // IWYU pragma: export

namespace mqt {
namespace py = pybind11;
using namespace py::literals;

} // namespace mqt
