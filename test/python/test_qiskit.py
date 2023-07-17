"""Test Qiskit import."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, Qubit
from qiskit.circuit.library import MCXRecursive, MCXVChain
from qiskit.transpiler import Layout, TranspileLayout

from mqt.core.qiskit_utils import qiskit_to_mqt


def test_empty_circuit() -> None:
    """Test import."""
    q = QuantumCircuit()

    mqt_qc = qiskit_to_mqt(q)
    assert mqt_qc.n_qubits == 0
    assert mqt_qc.n_ops == 0


def test_single_gate() -> None:
    """Test import."""
    q = QuantumCircuit(1)
    q.h(0)

    mqt_qc = qiskit_to_mqt(q)
    assert mqt_qc.n_qubits == 1
    assert mqt_qc.n_ops == 1
    assert mqt_qc[0].name.strip() == "h"
    assert mqt_qc[0].n_qubits == 1


def test_two_qubit_gate() -> None:
    """Test import of two qubit gate."""
    q = QuantumCircuit(2)
    q.cx(0, 1)
    mqt_qc = qiskit_to_mqt(q)
    assert mqt_qc.n_qubits == 2
    assert mqt_qc.n_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert mqt_qc[0].n_qubits == 2
    assert {control.qubit for control in mqt_qc[0].controls} == {0}


def test_mcx() -> None:
    """Test import of mcx gate."""
    q = QuantumCircuit(3)
    q.mcx([0, 1], 2)
    mqt_qc = qiskit_to_mqt(q)
    assert mqt_qc.n_qubits == 3
    assert mqt_qc.n_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert mqt_qc[0].n_qubits == 3
    assert {control.qubit for control in mqt_qc[0].controls} == {0, 1}


def test_mcx_recursive() -> None:
    """Test import of large mcx gate."""
    q = QuantumCircuit(9)
    q.append(MCXRecursive(num_ctrl_qubits=7), range(9))
    mqt_qc = qiskit_to_mqt(q)
    assert mqt_qc.n_qubits == 9
    assert mqt_qc.n_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert mqt_qc[0].n_qubits == 9
    assert {control.qubit for control in mqt_qc[0].controls} == {1, 2, 3, 4, 5, 6, 7}


def test_mcx_vchain() -> None:
    """Test import of mcx gate with v-chain."""
    q = QuantumCircuit(9)
    q.append(MCXVChain(num_ctrl_qubits=5), range(9))
    mqt_qc = qiskit_to_mqt(q)
    assert mqt_qc.n_qubits == 9
    assert mqt_qc.n_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert mqt_qc[0].n_qubits == 9
    assert {control.qubit for control in mqt_qc[0].controls} == {3, 4, 5, 6, 7}


def test_custom_gate() -> None:
    """Test import of custom gate."""
    custom_instr = QuantumCircuit(3)
    custom_instr.h(0)
    custom_instr.cx(0, 1)
    custom_instr.cx(0, 2)
    custom_instr = custom_instr.to_instruction()
    qc = QuantumCircuit(3)
    qc.append(custom_instr, range(3))
    mqt_qc = qiskit_to_mqt(qc)
    assert mqt_qc.n_qubits == 3
    assert mqt_qc.n_ops == 3
    assert mqt_qc[0].name.strip() == "h"
    assert mqt_qc[1].name.strip() == "x"
    assert mqt_qc[2].name.strip() == "x"
    assert mqt_qc[0].n_qubits == 3
    assert mqt_qc[1].n_qubits == 3
    assert mqt_qc[1].n_qubits == 3
    assert {control.qubit for control in mqt_qc[1].controls} == {0}
    assert {control.qubit for control in mqt_qc[2].controls} == {0}


def test_initial_layout() -> None:
    """Test import of initial layout."""
    qc = QuantumCircuit(3)
    q_reg = QuantumRegister(3, "q")
    qc._layout = TranspileLayout(  # noqa: SLF001
        Layout.from_intlist([2, 1, 0], q_reg), {Qubit(q_reg, 0): 0, Qubit(q_reg, 1): 1, Qubit(q_reg, 2): 2}
    )
    qc.h(0)
    qc.s(1)
    qc.x(2)
    mqt_qc = qiskit_to_mqt(qc)
    assert mqt_qc.n_qubits == 3
    assert mqt_qc.n_ops == 3
    assert mqt_qc.initial_layout.apply([0, 1, 2]) == [2, 1, 0]
