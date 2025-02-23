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
#include "dd/Package.hpp"
#include "python/pybind11.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

namespace {
/// Recursive helper function to create a vector DD from a numpy array
dd::vCachedEdge makeDDFromVector(
    dd::Package<>& p,
    const py::detail::unchecked_reference<std::complex<dd::fp>, 1>& v,
    const size_t startIdx, const size_t endIdx, const dd::Qubit level) {
  if (level == 0U) {
    const auto zeroSuccessor = dd::vCachedEdge::terminal(v(startIdx));
    const auto oneSuccessor = dd::vCachedEdge::terminal(v(startIdx + 1));
    return p.makeDDNode<dd::vNode, dd::CachedEdge>(
        0, {zeroSuccessor, oneSuccessor});
  }

  const auto half = startIdx + ((endIdx - startIdx) / 2);
  const auto zeroSuccessor = makeDDFromVector(p, v, startIdx, half, level - 1);
  const auto oneSuccessor = makeDDFromVector(p, v, half, endIdx, level - 1);
  return p.makeDDNode<dd::vNode, dd::CachedEdge>(level,
                                                 {zeroSuccessor, oneSuccessor});
}
} // namespace

void registerVectorDDs(const py::module& mod) {
  auto vec = py::class_<dd::vEdge>(mod, "VectorDD");

  ///------------------------------------------------------------------------///
  /// Constructors
  ///------------------------------------------------------------------------///

  vec.def_static(
      "zero_state",
      [](const size_t num_qubits, dd::Package<>& p) {
        return p.makeZeroState(num_qubits);
      },
      "num_qubits"_a, "dd_package"_a);

  vec.def_static(
      "computational_basis_state",
      [](const size_t num_qubits, const std::vector<bool>& state,
         dd::Package<>& p) { return p.makeBasisState(num_qubits, state); },
      "num_qubits"_a, "state"_a, "dd_package"_a);

  py::enum_<dd::BasisStates>(mod, "BasisStates")
      .value("zero", dd::BasisStates::zero)
      .value("one", dd::BasisStates::one)
      .value("plus", dd::BasisStates::plus)
      .value("minus", dd::BasisStates::minus)
      .value("right", dd::BasisStates::right)
      .value("left", dd::BasisStates::left);

  vec.def_static(
      "make_basis_state",
      [](const size_t num_qubits, const std::vector<dd::BasisStates>& state,
         dd::Package<>& p) { return p.makeBasisState(num_qubits, state); },
      "num_qubits"_a, "state"_a, "dd_package"_a);

  vec.def_static(
      "ghz_state",
      [](const size_t num_qubits, dd::Package<>& p) {
        return p.makeGHZState(num_qubits);
      },
      "num_qubits"_a, "dd_package"_a);

  vec.def_static(
      "w_state",
      [](const size_t num_qubits, dd::Package<>& p) {
        return p.makeWState(num_qubits);
      },
      "num_qubits"_a, "dd_package"_a);

  vec.def_static(
      "from_vector",
      [](const py::array_t<std::complex<dd::fp>>& v, dd::Package<>& p) {
        const auto data = v.unchecked<1>();
        const auto length = data.shape(0);
        if (length == 0) {
          return dd::vEdge::one();
        }
        if ((length & (length - 1)) != 0) {
          throw std::invalid_argument(
              "State vector must have a length of a power of two.");
        }
        if (length == 1) {
          return dd::vEdge::terminal(p.cn.lookup(data(0)));
        }

        const auto level = static_cast<dd::Qubit>(std::log2(length) - 1);
        const auto state = makeDDFromVector(p, data, 0, length, level);
        const dd::vEdge e{state.p, p.cn.lookup(state.w)};
        p.incRef(e);
        return e;
      },
      "state"_a, "dd_package"_a);

  ///------------------------------------------------------------------------///
  /// Reference Counting
  ///------------------------------------------------------------------------///
  vec.def(
      "inc_ref", [](const dd::vEdge& v, dd::Package<>& p) { p.incRef(v); },
      "dd_package"_a);
  vec.def(
      "dec_ref", [](const dd::vEdge& v, dd::Package<>& p) { p.decRef(v); },
      "dd_package"_a);
}

} // namespace mqt
