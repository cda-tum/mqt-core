#pragma once

#include "pybind11/pybind11.h"

namespace py = pybind11;

#include "QuantumComputation.hpp"

namespace qc::qiskit {
using namespace pybind11::literals;

class QasmQobjExperiment {
public:
  static void import(QuantumComputation& qc, const py::object& circ);

protected:
  static void emplaceInstruction(QuantumComputation& qc,
                                 const py::object& instruction);

  static void addOperation(QuantumComputation& qc, OpType type,
                           const py::list& qubits, const py::list& params);

  static void addTwoTargetOperation(QuantumComputation& qc, OpType type,
                                    const py::list& qubits,
                                    const py::list& params);
};
} // namespace qc::qiskit
