#pragma once

#include "pybind11/pybind11.h"

#include <pybind11/pytypes.h>
#include <regex>
#include <type_traits>
#include <variant>

namespace py = pybind11;

#include "QuantumComputation.hpp"

namespace qc::qiskit {
using namespace pybind11::literals;

class QuantumCircuit {
public:
  static void import(QuantumComputation& qc, const py::object& circ);

protected:
  static void emplaceOperation(QuantumComputation& qc,
                               const py::object& instruction,
                               const py::list& qargs, const py::list& cargs,
                               const py::list& params, const py::dict& qubitMap,
                               const py::dict& clbitMap);

  static SymbolOrNumber parseSymbolicExpr(const py::object& pyExpr);

  static SymbolOrNumber parseParam(const py::object& param);

  static void addOperation(QuantumComputation& qc, OpType type,
                           const py::list& qargs, const py::list& params,
                           const py::dict& qubitMap);

  static void addTwoTargetOperation(QuantumComputation& qc, OpType type,
                                    const py::list& qargs,
                                    const py::list& params,
                                    const py::dict& qubitMap);

  static void importDefinition(QuantumComputation& qc, const py::object& circ,
                               const py::list& qargs, const py::list& cargs,
                               const py::dict& qubitMap,
                               const py::dict& clbitMap);

  static void importInitialLayout(QuantumComputation& qc,
                                  const py::object& circ);
};
} // namespace qc::qiskit
