# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check the circuits and metric definitions used by compiler benchmarks."""

from __future__ import annotations

import os

import pytest
from benchmarks.compiler import compile_to_native, make_benchmark, make_target, native_depth
from qiskit import QuantumCircuit

from mqt.core.mlir import sample


def test_requested_cpu_affinity_must_be_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """An asv affinity failure must stop measurement instead of changing mapping trials."""
    monkeypatch.setenv("MQT_BENCHMARK_CPU", "2")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {2, 3}, raising=False)
    with pytest.raises(RuntimeError, match="CPU affinity was not applied"):
        make_target(4, "line")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {2})
    make_target(4, "line")


def test_depth_counts_parallel_native_gates_and_excludes_measurement() -> None:
    """Parallel native gates occupy one layer; terminal measurements add none."""
    circuit = QuantumCircuit(4, 4)
    circuit.u(0.1, 0.2, 0.3, 0)
    circuit.u(0.4, 0.5, 0.6, 2)
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    circuit.measure(range(4), range(4))
    assert native_depth(circuit) == 2


def test_depth_preserves_barrier_synchronization() -> None:
    """A barrier synchronizes layers without counting as another gate."""
    circuit = QuantumCircuit(2)
    circuit.u(0.1, 0.2, 0.3, 0)
    circuit.u(0.4, 0.5, 0.6, 0)
    circuit.barrier()
    circuit.u(0.7, 0.8, 0.9, 1)
    assert native_depth(circuit) == 3


def test_depth_rejects_non_native_output() -> None:
    """An undecomposed gate must not look like a cheap native operation."""
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    with pytest.raises(ValueError, match="Expected native U/CX"):
        native_depth(circuit)


def test_depth_rejects_nonterminal_measurement() -> None:
    """Dynamic circuits need a different depth definition."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    circuit.u(0.1, 0.2, 0.3, 0)
    with pytest.raises(ValueError, match="terminal measurements"):
        native_depth(circuit)


def test_depth_allows_gates_on_unmeasured_wires() -> None:
    """A final measurement on one wire does not finish unrelated wires."""
    circuit = QuantumCircuit(2, 1)
    circuit.u(0.1, 0.2, 0.3, 0)
    circuit.measure(0, 0)
    circuit.u(0.4, 0.5, 0.6, 1)
    assert native_depth(circuit) == 1


@pytest.mark.parametrize("topology", ["all_to_all", "line"])
@pytest.mark.parametrize("family", ["ghz", "qft", "bv"])
@pytest.mark.parametrize("qubits", [4, 8, 16])
def test_benchmark_circuits_compile_to_native_gates(family: str, topology: str, qubits: int) -> None:
    """The corpus produces observable, deterministic native output."""
    benchmark = make_benchmark(family, qubits)
    program = benchmark.generate()
    original = program.ir
    target = make_target(qubits, topology)
    first = compile_to_native(program, target)
    second = compile_to_native(program, target)
    circuit = first.to_qiskit(target=target.target)
    assert program.ir == original
    assert circuit.count_ops()["measure"] == benchmark.output.width
    assert first.num_two_qubit_gates() == circuit.count_ops().get("cx", 0)
    assert native_depth(circuit) > 0
    # QFT's reference support grows with the period exponent; bound sampling error.
    shots = max(4096, 256 * 2 ** (qubits // 2)) if family == "qft" else 4096
    evaluation = benchmark.evaluate(sample(first, shots=shots, seed=42))
    assert evaluation.total_variation_distance < 0.05
    assert first.num_two_qubit_gates() == second.num_two_qubit_gates()
    assert native_depth(circuit) == native_depth(second.to_qiskit(target=target.target))
    if topology == "line":
        for instruction in circuit.data:
            if instruction.operation.name == "cx":
                first_qubit, second_qubit = (circuit.find_bit(qubit).index for qubit in instruction.qubits)
                assert abs(first_qubit - second_qubit) == 1


def test_compiled_bell_circuit_metrics() -> None:
    """A compiled Bell preparation needs one CX and two native gate layers."""
    target = make_target(2, "line")
    compiled = compile_to_native(make_benchmark("ghz", 2).generate(), target)
    assert compiled.num_two_qubit_gates() == 1
    assert native_depth(compiled.to_qiskit(target=target.target)) == 2
