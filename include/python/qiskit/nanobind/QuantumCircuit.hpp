#pragma once

#include "QuantumComputation.hpp"
#include "python/nanobind.hpp"

#include <regex>
#include <type_traits>
#include <variant>

namespace qc::qiskit {
namespace nb = nanobind;
using namespace nb::literals;

class QuantumCircuit {
public:
  static void import(QuantumComputation& qc, const nb::object& circ);

protected:
  static void emplaceOperation(QuantumComputation& qc,
                               const nb::object& instruction,
                               const nb::list& qargs, const nb::list& cargs,
                               const nb::list& params, const nb::dict& qubitMap,
                               const nb::dict& clbitMap);

  static SymbolOrNumber parseSymbolicExpr(const nb::object& pyExpr);

  static SymbolOrNumber parseParam(const nb::object& param);

  static void addOperation(QuantumComputation& qc, OpType type,
                           const nb::list& qargs, const nb::list& params,
                           const nb::dict& qubitMap);

  static void addTwoTargetOperation(QuantumComputation& qc, OpType type,
                                    const nb::list& qargs,
                                    const nb::list& params,
                                    const nb::dict& qubitMap);

  static void importDefinition(QuantumComputation& qc, const nb::object& circ,
                               const nb::list& qargs, const nb::list& cargs,
                               const nb::dict& qubitMap,
                               const nb::dict& clbitMap);

  static void importInitialLayout(QuantumComputation& qc,
                                  const nb::object& circ);
};
} // namespace qc::qiskit
