"""Test Qiskit import."""

from __future__ import annotations

from qiskit import QuantumCircuit

from mqt.core.qiskit_utils import quantum_computation_from_qiskit_circuit


def test_import() -> None:
    """Test import."""
    q = QuantumCircuit(2)
    q.h(0)

    mqt_qc = quantum_computation_from_qiskit_circuit(q)
    assert mqt_qc.n_qubits == 2
