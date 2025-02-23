/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/CachedEdge.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/Operation.hpp"
#include "python/pybind11.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

using SingleQubitGateMatrix =
    std::array<std::array<std::complex<dd::fp>, 2>, 2>;

namespace {
/// Recursive helper function to create a matrix DD from a numpy array
dd::mCachedEdge makeDDFromMatrix(
    dd::Package<>& p,
    const py::detail::unchecked_reference<std::complex<dd::fp>, 2>& m,
    const size_t rowStart, const size_t rowEnd, const size_t colStart,
    const size_t colEnd, const dd::Qubit level) {
  if (level == 0U) {
    const auto zeroSuccessor = dd::mCachedEdge::terminal(m(rowStart, colStart));
    const auto oneSuccessor =
        dd::mCachedEdge::terminal(m(rowStart, colStart + 1));
    const auto twoSuccessor =
        dd::mCachedEdge::terminal(m(rowStart + 1, colStart));
    const auto threeSuccessor =
        dd::mCachedEdge::terminal(m(rowStart + 1, colStart + 1));
    return p.makeDDNode<dd::mNode, dd::CachedEdge>(
        0, {zeroSuccessor, oneSuccessor, twoSuccessor, threeSuccessor});
  }

  const auto rowHalf = rowStart + ((rowEnd - rowStart) / 2);
  const auto colHalf = colStart + ((colEnd - colStart) / 2);
  return p.makeDDNode<dd::mNode, dd::CachedEdge>(
      level,
      {makeDDFromMatrix(p, m, rowStart, rowHalf, colStart, colHalf, level - 1),
       makeDDFromMatrix(p, m, rowStart, rowHalf, colHalf, colEnd, level - 1),
       makeDDFromMatrix(p, m, rowHalf, rowEnd, colStart, colHalf, level - 1),
       makeDDFromMatrix(p, m, rowHalf, rowEnd, colHalf, colEnd, level - 1)});
}
} // namespace

void registerMatrixDDs(const py::module& mod) {
  auto mat = py::class_<dd::mEdge>(mod, "MatrixDD");

  ///------------------------------------------------------------------------///
  /// Constructors
  ///------------------------------------------------------------------------///

  mat.def_static(
      "identity", [](dd::Package<>& p) { return p.makeIdent(); },
      "dd_package"_a);

  mat.def_static(
      "single_qubit_gate",
      [](const SingleQubitGateMatrix& m, const dd::Qubit target,
         dd::Package<>& p) {
        return p.makeGateDD({m[0][0], m[0][1], m[1][0], m[1][1]}, target);
      },
      "matrix"_a, "target"_a, "dd_package"_a);

  mat.def_static(
      "controlled_single_qubit_gate",
      [](const SingleQubitGateMatrix& m, const qc::Control& control,
         const dd::Qubit target, dd::Package<>& p) {
        return p.makeGateDD({m[0][0], m[0][1], m[1][0], m[1][1]}, control,
                            target);
      },
      "matrix"_a, "control"_a, "target"_a, "dd_package"_a);

  mat.def_static(
      "multi_controlled_single_qubit_gate",
      [](const SingleQubitGateMatrix& m, const qc::Controls& controls,
         const dd::Qubit target, dd::Package<>& p) {
        return p.makeGateDD({m[0][0], m[0][1], m[1][0], m[1][1]}, controls,
                            target);
      },
      "matrix"_a, "controls"_a, "target"_a, "dd_package"_a);

  mat.def_static(
      "two_qubit_gate",
      [](const dd::TwoQubitGateMatrix& m, const dd::Qubit target0,
         const dd::Qubit target1, dd::Package<>& p) {
        return p.makeTwoQubitGateDD(m, target0, target1);
      },
      "matrix"_a, "target0"_a, "target1"_a, "dd_package"_a);

  mat.def_static(
      "controlled_two_qubit_gate",
      [](const dd::TwoQubitGateMatrix& m, const qc::Control& control,
         const dd::Qubit target0, const dd::Qubit target1, dd::Package<>& p) {
        return p.makeTwoQubitGateDD(m, control, target0, target1);
      },
      "matrix"_a, "control"_a, "target0"_a, "target1"_a, "dd_package"_a);

  mat.def_static(
      "multi_controlled_two_qubit_gate",
      [](const dd::TwoQubitGateMatrix& m, const qc::Controls& controls,
         const dd::Qubit target0, const dd::Qubit target1, dd::Package<>& p) {
        return p.makeTwoQubitGateDD(m, controls, target0, target1);
      },
      "matrix"_a, "controls"_a, "target0"_a, "target1"_a, "dd_package"_a);

  mat.def_static(
      "from_matrix",
      [](const py::array_t<std::complex<dd::fp>>& m, dd::Package<>& p) {
        const auto data = m.unchecked<2>();
        const auto rows = static_cast<size_t>(data.shape(0));
        const auto cols = static_cast<size_t>(data.shape(1));
        if (rows != cols) {
          throw std::invalid_argument("Matrix must be square.");
        }
        if (rows == 0) {
          return dd::mEdge::one();
        }
        if ((rows & (rows - 1)) != 0) {
          throw std::invalid_argument(
              "Matrix must have a size of a power of two.");
        }
        if (rows == 1) {
          return dd::mEdge::terminal(p.cn.lookup(data(0, 0)));
        }
        const auto level = static_cast<dd::Qubit>(std::log2(rows) - 1);
        const auto matrixDD =
            makeDDFromMatrix(p, data, 0, rows, 0, cols, level);
        return dd::mEdge{matrixDD.p, p.cn.lookup(matrixDD.w)};
      },
      "matrix"_a, "dd_package"_a);

  mat.def_static(
      "from_operation",
      [](const qc::Operation& op, dd::Package<>& p, const bool invert = false) {
        if (invert) {
          return dd::getInverseDD(op, p);
        }
        return dd::getDD(op, p);
      },
      "operation"_a, "dd_package"_a, "invert"_a = false);

  ///------------------------------------------------------------------------///
  /// Reference Counting
  ///------------------------------------------------------------------------///
  mat.def(
      "inc_ref", [](const dd::mEdge& m, dd::Package<>& p) { p.incRef(m); },
      "dd_package"_a);
  mat.def(
      "dec_ref", [](const dd::mEdge& m, dd::Package<>& p) { p.decRef(m); },
      "dd_package"_a);
}
} // namespace mqt
