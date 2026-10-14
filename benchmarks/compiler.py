# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Benchmark native compilation of the structured MQT Core circuit catalog."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mqt.core.bench import bv, ghz, qft
from mqt.core.mlir import (
    CompilerTarget,
    PayloadEncoding,
    PayloadFormat,
    PayloadSpecification,
    QCProgram,
    TargetEnvironment,
)

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

PARAMS = (("ghz", "qft", "bv"), (4, 8, 16), ("all_to_all", "line"))
PARAM_NAMES = ("circuit", "qubits", "topology")
VERSION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def make_benchmark(family: str, num_qubits: int) -> ghz.GHZ | qft.QFT | bv.BV:
    """Select a static Core benchmark using the requested total qubit budget.

    Returns:
        A typed Core benchmark with its canonical specification and reference.

    Raises:
        ValueError: If the benchmark family is unsupported.
    """
    if family == "ghz":
        return ghz.GHZ(ghz.Options(qubits=num_qubits, topology=ghz.Topology.STAR, basis=ghz.Basis.Z))
    if family == "qft":
        return qft.QFT(qft.Options(qubits=num_qubits, period_exponent=num_qubits // 2, method=qft.Method.STANDARD))
    if family == "bv":
        return bv.BV(bv.Options(hidden_bitstring="1" * (num_qubits - 1), method=bv.Method.STATIC))
    msg = f"Unknown benchmark circuit: {family}"
    raise ValueError(msg)


def make_target(num_qubits: int, topology: str) -> TargetEnvironment:
    """Create a fixed U/CX target with a static Base QIR payload contract.

    Returns:
        A target environment that requires static quantum loops to be unrolled.

    Raises:
        ValueError: If the topology is unsupported.
        RuntimeError: If a requested CPU affinity was not applied.
    """
    if requested_cpu := os.environ.get("MQT_BENCHMARK_CPU"):
        affinity = getattr(os, "sched_getaffinity", None)
        if affinity is None or affinity(0) != {int(requested_cpu)}:
            msg = "Benchmark CPU affinity was not applied; refusing uncontrolled mapping trials."
            raise RuntimeError(msg)
    if topology not in {"all_to_all", "line"}:
        msg = f"Unknown benchmark topology: {topology}"
        raise ValueError(msg)
    connectivity = (
        CompilerTarget.Connectivity([(qubit, qubit + 1) for qubit in range(num_qubits - 1)])
        if topology == "line"
        else CompilerTarget.Connectivity.all_to_all()
    )
    target = CompilerTarget(
        num_qubits,
        connectivity=connectivity,
        native_operations=CompilerTarget.NativeOperations([
            CompilerTarget.OperationCapability("u", 1, 3),
            CompilerTarget.OperationCapability("cx", 2, 0),
            CompilerTarget.OperationCapability("measure", 1, 0),
        ]),
    )
    payload = PayloadSpecification(
        PayloadFormat("qir", "2.1.0", "base", PayloadEncoding.BINARY),
        optional_capabilities_known=True,
    )
    return TargetEnvironment(target, payload)


def compile_to_native(program: QCProgram, environment: TargetEnvironment) -> QCProgram:
    """Compile an independent copy through QCO and return native QC output.

    Returns:
        The circuit after target optimization, mapping, and native synthesis.
    """
    qco = program.to_qco(copy=True)
    qco.compile_for_target(environment)
    return qco.to_qc()


def native_depth(circuit: QuantumCircuit) -> int:
    """Measure unweighted native gate depth, with barriers as synchronization.

    Returns:
        The U/CX circuit depth, excluding terminal measurements.

    Raises:
        ValueError: If output contains other gates or nonterminal measurements.
    """
    measured = set()
    for instruction in circuit.data:
        name = instruction.operation.name
        if name not in {"u", "cx", "measure", "barrier"}:
            msg = f"Expected native U/CX output, found {name}."
            raise ValueError(msg)
        if name in {"u", "cx"} and not measured.isdisjoint(instruction.qubits):
            msg = "Depth benchmarks require terminal measurements."
            raise ValueError(msg)
        if name == "measure":
            measured.update(instruction.qubits)
    return circuit.depth(lambda instruction: instruction.operation.name in {"u", "cx"})


class Compilation:
    """Time input copying, target compilation, and result disposal."""

    params: ClassVar = PARAMS
    param_names: ClassVar = PARAM_NAMES
    version = VERSION
    rounds = 2
    repeat = 3
    number = 1
    warmup_time = 0
    timeout = 120

    def setup(self, circuit: str, qubits: int, topology: str) -> None:
        """Generate a Core program and target outside the timed region."""
        self.program = make_benchmark(circuit, qubits).generate()
        self.target = make_target(qubits, topology)

    def time_compile(self, _circuit: str, _qubits: int, _topology: str) -> None:
        """Compile a fresh copy through the canonical target pipeline."""
        compile_to_native(self.program, self.target)


class CircuitQuality:
    """Track two-qubit gate count and depth after mapping and native synthesis."""

    params: ClassVar = PARAMS
    param_names: ClassVar = PARAM_NAMES
    version = VERSION
    unit = "count"
    timeout = 120

    def setup(self, circuit: str, qubits: int, topology: str) -> None:
        """Compile and analyze the same output outside timing benchmarks."""
        target = make_target(qubits, topology)
        compiled = compile_to_native(make_benchmark(circuit, qubits).generate(), target)
        self.two_qubit_count = compiled.num_two_qubit_gates()
        self.depth = native_depth(compiled.to_qiskit(target=target.target))

    def track_two_qubit_count(self, _circuit: str, _qubits: int, _topology: str) -> int:
        """Return the number of native two-qubit gates."""
        return self.two_qubit_count

    def track_depth(self, _circuit: str, _qubits: int, _topology: str) -> int:
        """Return native gate depth without terminal measurements."""
        return self.depth
