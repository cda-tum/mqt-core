"""Test Qiskit import."""

from __future__ import annotations

from typing import cast

from qiskit import QuantumCircuit
from qiskit.circuit import AncillaRegister, ClassicalRegister, Parameter, QuantumRegister, Qubit
from qiskit.circuit.library import MCXRecursive, MCXVChain, XXMinusYYGate, XXPlusYYGate
from qiskit.transpiler import Layout, TranspileLayout

from mqt.core import Expression, SymbolicOperation
from mqt.core.qiskit import qiskit_to_mqt

import pdb

def test_empty_circuit() -> None:
    """Test import of empty circuit."""
    q = QuantumCircuit()

    mqt_qc = qiskit_to_mqt(q)
    assert mqt_qc.n_qubits == 0
    assert mqt_qc.n_ops == 0


def test_single_gate() -> None:
    """Test import of single-gate circuit."""
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
    assert mqt_qc.n_ops == 1
    assert mqt_qc[0][0].name.strip() == "h"
    assert mqt_qc[0][1].name.strip() == "x"
    assert mqt_qc[0][2].name.strip() == "x"
    assert mqt_qc[0][0].n_qubits == 3
    assert mqt_qc[0][1].n_qubits == 3
    assert mqt_qc[0][1].n_qubits == 3
    assert {control.qubit for control in mqt_qc[0][1].controls} == {0}
    assert {control.qubit for control in mqt_qc[0][2].controls} == {0}


def test_layout() -> None:
    """Test import of initial layout."""
    qc = QuantumCircuit(3)
    q_reg = QuantumRegister(3, "q")
    qc._layout = TranspileLayout(  # noqa: SLF001
        Layout.from_intlist([2, 1, 0], q_reg),
        {Qubit(q_reg, 0): 0, Qubit(q_reg, 1): 1, Qubit(q_reg, 2): 2},
        Layout.from_intlist([1, 2, 0], q_reg),
    )
    qc.h(0)
    qc.s(1)
    qc.x(2)
    mqt_qc = qiskit_to_mqt(qc)
    assert mqt_qc.n_qubits == 3
    assert mqt_qc.n_ops == 3
    assert mqt_qc.initial_layout.apply([0, 1, 2]) == [2, 1, 0]
    assert mqt_qc.output_permutation.apply([0, 2, 1]) == [2, 1, 0]


def test_ancilla() -> None:
    """Test import of ancilla register."""
    anc_reg = AncillaRegister(1, "anc")
    q_reg = QuantumRegister(1, "q")
    qc = QuantumCircuit(q_reg, anc_reg)
    qc.h(anc_reg[0])
    qc.cx(anc_reg[0], q_reg[0])
    mqt_qc = qiskit_to_mqt(qc)
    assert mqt_qc.n_qubits_without_ancillae == 1
    assert mqt_qc.n_ancillae == 1


def test_classical() -> None:
    """Test import of classical register."""
    c_reg = ClassicalRegister(1, "c")
    q_reg = QuantumRegister(1, "q")
    qc = QuantumCircuit(q_reg, c_reg)
    qc.h(q_reg[0])
    qc.barrier(q_reg[0])
    qc.measure(q_reg[0], c_reg[0])
    qc.reset(q_reg[0])
    mqt_qc = qiskit_to_mqt(qc)
    assert mqt_qc.n_qubits == 1
    assert mqt_qc.n_cbits == 1


def test_operations() -> None:
    """Test import of operations."""
    qc = QuantumCircuit(3)
    qc.i(0)  # op 1
    qc.cy(0, 1)  # op 2
    qc.z(2)  # op 3
    qc.sdg(2)  # op 4
    qc.t(2)  # op 5
    qc.tdg(1)  # op 6
    qc.crx(0.5, 2, 0)  # op 7
    qc.ry(0.5, 0)  # op 8
    qc.rz(0.5, 1)  # op 9
    qc.mcp(0.5, [0, 1], 2)  # op 10
    qc.sx(0)  # op 11
    qc.swap(0, 1)  # op 12
    qc.iswap(0, 1)  # op 13
    qc.dcx(0, 1)  # op 14
    qc.ecr(0, 1)  # op 15
    qc.rxx(0.5, 0, 1)  # op 16
    qc.rzz(0.5, 0, 1)  # op 17
    qc.ryy(0.5, 0, 1)  # op 18
    qc.append(XXMinusYYGate(0.1), [0, 1])  # op 19
    qc.append(XXPlusYYGate(0.1), [0, 1])  # op 20

    mqt_qc = qiskit_to_mqt(qc)
    assert mqt_qc.n_qubits == 3
    assert mqt_qc.n_ops == 20
    assert not mqt_qc.is_variable_free()


def test_symbolic() -> None:
    """Test import of symbolic parameters."""
    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc.rx(2 * theta + phi, 0)
    mqt_qc = qiskit_to_mqt(qc)

    assert mqt_qc.n_qubits == 1
    assert mqt_qc.n_ops == 1
    assert mqt_qc[0].name.strip() == "rx"
    assert isinstance(mqt_qc[0], SymbolicOperation)
    assert isinstance(mqt_qc[0].get_parameter(0), Expression)
    expr = cast(Expression, mqt_qc[0].get_parameter(0))
    assert expr.num_terms() == 2
    assert expr.constant == 0.0
