"""Test Qiskit import."""

from __future__ import annotations

from typing import cast

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import AncillaRegister, ClassicalRegister, Parameter, QuantumRegister
from qiskit.circuit.library import U2Gate, XXMinusYYGate, XXPlusYYGate
from qiskit.providers.fake_provider import Fake5QV1

from mqt.core.ir.operations import CompoundOperation, SymbolicOperation
from mqt.core.ir.symbolic import Expression
from mqt.core.plugins.qiskit import qiskit_to_mqt


def test_empty_circuit() -> None:
    """Test import of empty circuit."""
    q = QuantumCircuit()

    mqt_qc = qiskit_to_mqt(q)
    print(mqt_qc)

    assert mqt_qc.num_qubits == 0
    assert mqt_qc.num_ops == 0


def test_single_gate() -> None:
    """Test import of single-gate circuit."""
    q = QuantumCircuit(1)
    q.h(0)
    mqt_qc = qiskit_to_mqt(q)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 1
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "h"


def test_two_qubit_gate() -> None:
    """Test import of two qubit gate."""
    q = QuantumCircuit(2)
    q.cx(0, 1)
    mqt_qc = qiskit_to_mqt(q)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 2
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert {control.qubit for control in mqt_qc[0].controls} == {0}


def test_mcx() -> None:
    """Test import of mcx gate."""
    q = QuantumCircuit(3)
    q.mcx([0, 1], 2)
    mqt_qc = qiskit_to_mqt(q)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 3
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert {control.qubit for control in mqt_qc[0].controls} == {0, 1}


def test_mcx_recursive() -> None:
    """Test import of large mcx gate."""
    q = QuantumCircuit(9)
    q.mcx(control_qubits=list(range(7)), target_qubit=7, ancilla_qubits=list(range(8, 9)), mode="recursion")
    mqt_qc = qiskit_to_mqt(q)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 9
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert {control.qubit for control in mqt_qc[0].controls} == {0, 1, 2, 3, 4, 5, 6}
    assert not mqt_qc[0].acts_on(8)


def test_small_mcx_recursive() -> None:
    """Test import of small mcx_recursive gate."""
    q = QuantumCircuit(5)
    q.mcx(target_qubit=4, control_qubits=list(range(4)), mode="recursion")
    mqt_qc = qiskit_to_mqt(q)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 5
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert {control.qubit for control in mqt_qc[0].controls} == {0, 1, 2, 3}


def test_mcx_vchain() -> None:
    """Test import of mcx gate with v-chain."""
    q = QuantumCircuit(9)
    q.mcx(target_qubit=5, control_qubits=list(range(5)), ancilla_qubits=list(range(6, 9)), mode="v-chain")
    mqt_qc = qiskit_to_mqt(q)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 9
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert {control.qubit for control in mqt_qc[0].controls} == {0, 1, 2, 3, 4}
    for i in range(6, 9):
        assert not mqt_qc[0].acts_on(i)


def test_custom_gate() -> None:
    """Test import of custom gate."""
    custom_instr = QuantumCircuit(3, 1)
    custom_instr.h(0)
    custom_instr.cx(0, 1)
    custom_instr.cx(0, 2)
    custom_instr.measure(0, 0)
    custom_instr = custom_instr.to_instruction()
    qc = QuantumCircuit(3, 1)
    qc.append(custom_instr, range(3), range(1))
    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 3
    assert mqt_qc.num_ops == 1
    assert isinstance(mqt_qc[0], CompoundOperation)
    assert mqt_qc[0][0].name.strip() == "h"
    assert mqt_qc[0][1].name.strip() == "x"
    assert mqt_qc[0][2].name.strip() == "x"
    assert mqt_qc[0][3].name.strip() == "measure"
    assert {control.qubit for control in mqt_qc[0][1].controls} == {0}
    assert {control.qubit for control in mqt_qc[0][2].controls} == {0}


def test_ancilla() -> None:
    """Test import of ancilla register."""
    anc_reg = AncillaRegister(1, "anc")
    q_reg = QuantumRegister(1, "q")
    qc = QuantumCircuit(q_reg, anc_reg)
    qc.h(anc_reg[0])
    qc.cx(anc_reg[0], q_reg[0])
    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_data_qubits == 1
    assert mqt_qc.num_ancilla_qubits == 1


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
    print(mqt_qc)
    assert mqt_qc.num_qubits == 1
    assert mqt_qc.num_classical_bits == 1


def test_operations() -> None:
    """Test import of operations."""
    qc = QuantumCircuit(3)
    qc.id(0)
    qc.cy(0, 1)
    qc.z(2)
    qc.s(2)
    qc.sdg(2)
    qc.t(2)
    qc.tdg(1)
    qc.u(0.5, 0.5, 0.5, 0)
    qc.crx(0.5, 2, 0)
    qc.ry(0.5, 0)
    qc.rz(0.5, 1)
    qc.mcp(0.5, [0, 1], 2)
    qc.sx(0)
    qc.swap(0, 1)
    qc.iswap(0, 1)
    qc.dcx(0, 1)
    qc.ecr(0, 1)
    qc.rxx(0.5, 0, 1)
    qc.rzz(0.5, 0, 1)
    qc.ryy(0.5, 0, 1)
    qc.rzx(0.5, 0, 1)
    qc.append(U2Gate(0.5, 0.5), [0])
    qc.append(XXMinusYYGate(0.1, 0.0), [0, 1])
    qc.append(XXPlusYYGate(0.1, 0.0), [0, 1])

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 3
    assert mqt_qc.num_ops == 24
    assert mqt_qc.is_variable_free()


def test_symbolic() -> None:
    """Test import of symbolic parameters."""
    qc = QuantumCircuit(1)
    lambda_ = Parameter("lambda")
    phi = Parameter("phi")
    theta = Parameter("theta")
    qc.rx(2 * theta + phi / 2 - lambda_ + 2, 0)
    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)

    assert mqt_qc.num_qubits == 1
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "rx"
    assert isinstance(mqt_qc[0], SymbolicOperation)
    assert isinstance(mqt_qc[0].get_parameter(0), Expression)
    expr = cast(Expression, mqt_qc[0].get_parameter(0))
    print(expr)
    assert expr.num_terms() == 3
    assert expr.terms[0].coefficient == -1
    assert expr.terms[0].variable.name == "lambda"
    assert expr.terms[1].coefficient == 0.5
    assert expr.terms[1].variable.name == "phi"
    assert expr.terms[2].coefficient == 2
    assert expr.terms[2].variable.name == "theta"
    assert expr.constant == 2
    assert not mqt_qc.is_variable_free()

    qc = qc.assign_parameters({lambda_: 0, phi: 0, theta: 0})
    mqt_qc = qiskit_to_mqt(qc)
    assert mqt_qc.is_variable_free()
    assert mqt_qc[0].parameter[0] == 2


def test_symbolic_two_qubit() -> None:
    """Test import of symbolic two-qubit gate."""
    qc = QuantumCircuit(2)
    theta = Parameter("theta")
    qc.rxx(theta, 0, 1)
    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)

    assert mqt_qc.num_qubits == 2
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "rxx"
    assert isinstance(mqt_qc[0], SymbolicOperation)
    assert isinstance(mqt_qc[0].get_parameter(0), Expression)
    expr = cast(Expression, mqt_qc[0].get_parameter(0))
    assert expr.num_terms() == 1
    assert expr.constant == 0
    assert not mqt_qc.is_variable_free()


def test_trivial_initial_layout_multiple_registers() -> None:
    """Test that trivial initial layout import works with multiple registers.

    Correctly inferring the initial layout is not an easy task; especially when
    multiple registers are involved. This test checks that the initial layout
    is imported properly from a circuit with multiple registers that are not
    sorted alphabetically.
    """
    a = QuantumRegister(2, "a")
    b = QuantumRegister(2, "b")
    qc = QuantumCircuit(b, a)
    initial_layout = [0, 1, 2, 3]
    qc_transpiled = transpile(qc, initial_layout=initial_layout)
    mqt_qc = qiskit_to_mqt(qc_transpiled)
    for k, v in [(0, 0), (1, 1), (2, 2), (3, 3)]:
        assert mqt_qc.initial_layout[k] == v


def test_non_trivial_initial_layout_multiple_registers() -> None:
    """Test that non-trivial initial layout import works with multiple registers."""
    a = QuantumRegister(2, "a")
    b = QuantumRegister(2, "b")
    qc = QuantumCircuit(b, a)
    initial_layout = [3, 2, 1, 0]
    qc_transpiled = transpile(qc, initial_layout=initial_layout)
    mqt_qc = qiskit_to_mqt(qc_transpiled)
    for k, v in [(0, 3), (1, 2), (2, 1), (3, 0)]:
        assert mqt_qc.initial_layout[k] == v
        assert mqt_qc.output_permutation[k] == v


def test_non_symmetric_initial_layout_multiple_registers() -> None:
    """Test that non-symmetric initial layout import works with multiple registers."""
    a = QuantumRegister(2, "a")
    b = QuantumRegister(1, "b")
    qc = QuantumCircuit(b, a)
    initial_layout = [1, 2, 0]
    qc_transpiled = transpile(qc, initial_layout=initial_layout)
    mqt_qc = qiskit_to_mqt(qc_transpiled)
    for k, v in [(0, 2), (1, 0), (2, 1)]:
        assert mqt_qc.initial_layout[k] == v
        assert mqt_qc.output_permutation[k] == v


def test_initial_layout_with_ancilla_in_front() -> None:
    """Test that initial layout import works with ancilla in front."""
    a = QuantumRegister(2, "a")
    b_anc = AncillaRegister(1, "b")
    qc = QuantumCircuit(b_anc, a)
    initial_layout = [0, 1, 2]
    qc_transpiled = transpile(qc, initial_layout=initial_layout)
    mqt_qc = qiskit_to_mqt(qc_transpiled)
    print(mqt_qc)
    for k, v in [(0, 0), (1, 1), (2, 2)]:
        assert mqt_qc.initial_layout[k] == v
    assert mqt_qc.num_ancilla_qubits == 1
    assert mqt_qc.is_circuit_qubit_ancillary(0)


def test_initial_layout_with_ancilla_in_back() -> None:
    """Test that initial layout import works with ancilla in back."""
    a = QuantumRegister(2, "a")
    b_anc = AncillaRegister(1, "b")
    qc = QuantumCircuit(a, b_anc)
    initial_layout = [0, 1, 2]
    qc_transpiled = transpile(qc, initial_layout=initial_layout)
    mqt_qc = qiskit_to_mqt(qc_transpiled)
    print(mqt_qc)
    for k, v in [(0, 0), (1, 1), (2, 2)]:
        assert mqt_qc.initial_layout[k] == v
    assert mqt_qc.num_ancilla_qubits == 1
    assert mqt_qc.is_circuit_qubit_ancillary(2)


def test_symbolic_global_phase() -> None:
    """Test whether symbolic global phase works properly."""
    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    qc.global_phase = theta

    with pytest.warns(RuntimeWarning):
        mqt_qc = qiskit_to_mqt(qc)

    assert mqt_qc.global_phase == 0


def test_final_layout_without_permutation() -> None:
    """Test that the output permutation remains the same as the initial layout when routing is not performed."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    initial_layout = [1, 2, 0]
    seed = 123
    qc_transpiled = transpile(qc, initial_layout=initial_layout, seed_transpiler=seed)
    mqt_qc = qiskit_to_mqt(qc_transpiled)
    assert mqt_qc.initial_layout == {0: 2, 1: 0, 2: 1}
    assert mqt_qc.output_permutation == mqt_qc.initial_layout


def test_final_layout_with_permutation() -> None:
    """Test that the output permutation gets updated correctly when routing is performed."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(1, 0)
    qc.cx(1, 2)
    qc.measure_all()
    initial_layout = [1, 0, 3]
    seed = 123
    backend = Fake5QV1()
    qc_transpiled = transpile(qc, backend, initial_layout=initial_layout, seed_transpiler=seed)
    final_index_layout = qc_transpiled.layout.final_index_layout()
    mqt_qc = qiskit_to_mqt(qc_transpiled)
    # Check initialize_io_mapping doesn't change the final_layout
    assert mqt_qc.output_permutation == dict(enumerate(final_index_layout))


def test_final_layout_with_permutation_ancilla_in_front_and_back() -> None:
    """Test that permutation update is correct with multiple registers and ancilla qubits."""
    e = QuantumRegister(2, "e")
    f_anc = AncillaRegister(1, "f")
    b_anc = AncillaRegister(2, "b")
    qc = QuantumCircuit(f_anc, e, b_anc)
    qc.h(0)
    qc.cx(1, 0)
    qc.cx(1, 2)
    qc.measure_all()
    initial_layout = [1, 0, 3, 2, 4]
    seed = 123
    backend = Fake5QV1()
    qc_transpiled = transpile(qc, backend, initial_layout=initial_layout, seed_transpiler=seed)
    routing_permutation = qc_transpiled.layout.routing_permutation()
    mqt_qc = qiskit_to_mqt(qc_transpiled)

    # Check that output_permutation matches the result of applying the routing permutation to input_layout
    assert mqt_qc.output_permutation == {idx: routing_permutation[key] for idx, key in enumerate(initial_layout)}
