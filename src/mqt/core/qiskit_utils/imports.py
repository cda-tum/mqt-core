"""Functionality for interoperability with Qiskit."""
from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, cast

from qiskit.circuit import AncillaQubit, AncillaRegister, Clbit, Instruction, ParameterExpression, Qubit

from mqt.core._core import (
    Control,
    Expression,
    OpType,
    QuantumComputation,
    StandardOperation,
    SymbolicOperation,
    Term,
    Variable,
)

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import Layout


def qiskit_to_mqt(qiskit_circuit: QuantumCircuit) -> QuantumComputation:
    """Convert a Qiskit circuit to a QuantumComputation object.

    Args:
        qiskit_circuit (QuantumCircuit): The Qiskit circuit to convert.

    Returns:
        QuantumComputation: The converted circuit.
    """
    mqt_computation = QuantumComputation()

    if qiskit_circuit.name is not None:
        mqt_computation.name = qiskit_circuit.name

    qubit_index = 0
    qubit_map = {}
    qiskit_qregs = qiskit_circuit.qregs
    for reg in qiskit_qregs:
        size = reg.size
        name = reg.name
        if isinstance(reg, AncillaRegister):
            mqt_computation.add_ancillary_register(size, name)
            for i in range(size):
                qubit_map[AncillaQubit(reg, i)] = qubit_index
                qubit_index += 1
        else:
            mqt_computation.add_qubit_register(size, name)
            for i in range(size):
                qubit_map[Qubit(reg, i)] = qubit_index
                qubit_index += 1

    clbit_index = 0
    clbit_map = {}
    qiskit_cregs = qiskit_circuit.cregs
    for reg in qiskit_cregs:
        size = reg.size
        name = reg.name
        mqt_computation.add_classical_bit_register(size, name)
        for i in range(size):
            clbit_map[Clbit(reg, i)] = clbit_index
            clbit_index += 1

    mqt_computation.gphase = qiskit_circuit.global_phase

    data = qiskit_circuit.data
    for inst in data:
        instruction = inst[0]
        qargs = inst[1]
        cargs = inst[2]
        params = instruction.params

        _emplace_operation(mqt_computation, instruction, qargs, cargs, params, qubit_map, clbit_map)

    # import initial layout and output permutation in case it is available
    if qiskit_circuit.layout is not None:
        _import_layouts(mqt_computation, qiskit_circuit)

    mqt_computation.initialize_io_mapping()
    return mqt_computation


def _emplace_operation(
    mqt_computation: QuantumComputation,
    instr: Instruction,
    qargs: tuple[Qubit] | list[Qubit],
    cargs: tuple[Clbit] | list[Clbit],
    params: list[float | ParameterExpression],
    qubit_map: dict[Qubit, int],
    clbit_map: dict[Clbit, int],
) -> None:
    name = instr.name

    if name == "measure":
        ctrl = qubit_map[qargs[0]]
        target = clbit_map[cargs[0]]
        mqt_computation.measure(ctrl, target)
    elif name == "barrier":
        targets = [qubit_map[qubit] for qubit in qargs]
        mqt_computation.barrier(targets)
    elif name == "reset":
        targets = [qubit_map[qubit] for qubit in qargs]
        mqt_computation.reset(targets)
    elif name in ["i", "id", "iden"]:
        _add_operation(mqt_computation, OpType.i, qargs, params, qubit_map)
    elif name in ["x", "cx", "ccx", "ccx", "mcx", "mcx_gray"]:
        _add_operation(mqt_computation, OpType.x, qargs, params, qubit_map)
    elif name in ["y", "cy"]:
        _add_operation(mqt_computation, OpType.y, qargs, params, qubit_map)
    elif name in ["z", "cz"]:
        _add_operation(mqt_computation, OpType.z, qargs, params, qubit_map)
    elif name in ["h", "ch"]:
        _add_operation(mqt_computation, OpType.h, qargs, params, qubit_map)
    elif name == "s":
        _add_operation(mqt_computation, OpType.s, qargs, params, qubit_map)
    elif name == "sdg":
        _add_operation(mqt_computation, OpType.sdag, qargs, params, qubit_map)
    elif name == "t":
        _add_operation(mqt_computation, OpType.t, qargs, params, qubit_map)
    elif name == "tdg":
        _add_operation(mqt_computation, OpType.tdag, qargs, params, qubit_map)
    elif name in ["rx", "crx", "mcrx"]:
        _add_operation(mqt_computation, OpType.rx, qargs, params, qubit_map)
    elif name in ["ry", "cry", "mcry"]:
        _add_operation(mqt_computation, OpType.ry, qargs, params, qubit_map)
    elif name in ["rz", "crz", "mcrz"]:
        _add_operation(mqt_computation, OpType.rz, qargs, params, qubit_map)
    elif name in ["p", "u1", "cp", "cu1", "mcphase"]:
        _add_operation(mqt_computation, OpType.phase, qargs, params, qubit_map)
    elif name in ["sx", "csx"]:
        _add_operation(mqt_computation, OpType.sx, qargs, params, qubit_map)
    elif name in ["swap", "cswap"]:
        _add_two_target_operation(mqt_computation, OpType.swap, qargs, params, qubit_map)
    elif name == "iswap":
        _add_two_target_operation(mqt_computation, OpType.iswap, qargs, params, qubit_map)
    elif name == "dcx":
        _add_two_target_operation(mqt_computation, OpType.dcx, qargs, params, qubit_map)
    elif name == "ecr":
        _add_two_target_operation(mqt_computation, OpType.ecr, qargs, params, qubit_map)
    elif name == "rxx":
        _add_two_target_operation(mqt_computation, OpType.rxx, qargs, params, qubit_map)
    elif name == "ryy":
        _add_two_target_operation(mqt_computation, OpType.ryy, qargs, params, qubit_map)
    elif name == "rzz":
        _add_two_target_operation(mqt_computation, OpType.rzz, qargs, params, qubit_map)
    elif name == "xx_minus_yy":
        _add_two_target_operation(mqt_computation, OpType.xx_minus_yy, qargs, params, qubit_map)
    elif name == "xx_plus_yy":
        _add_two_target_operation(mqt_computation, OpType.xx_plus_yy, qargs, params, qubit_map)
    elif name == "mcx_recursive":
        if len(qargs) <= 5:
            _add_operation(mqt_computation, OpType.x, qargs, params, qubit_map)
        else:
            _add_operation(mqt_computation, OpType.x, list(qargs[1:]), params, qubit_map)
    elif name == "mcx_vchain":
        size = len(qargs)
        n_controls = (size + 1) // 2
        _add_operation(mqt_computation, OpType.x, list(qargs[n_controls - 2 :]), params, qubit_map)
    else:
        try:
            _import_definition(mqt_computation, instr.definition, qargs, cargs, qubit_map, clbit_map)
        except Exception:
            print("Failed to import instruction: " + name + " from Qiskit QuantumCircuit\n", sys.stderr)


_SUM_REGEX = re.compile("[+|-]?[^+-]+")
_PROD_REGEX = re.compile("[\\*/]?[^\\*/]+")


def _parse_symbolic_expression(qiskit_expr: ParameterExpression | float) -> float | Expression:
    if isinstance(qiskit_expr, float):
        return qiskit_expr

    expr_str = qiskit_expr.__str__().strip()
    expr = Expression()
    is_const = False
    for summand in _SUM_REGEX.findall(expr_str):
        sign = 1
        summand_no_operaror = summand
        if summand[0] == "+":
            summand_no_operaror = summand[1:]
        elif summand[0] == "-":
            summand_no_operaror = summand[1:]
            sign = -1

        coeff = 1.0
        var = ""
        for factor in _PROD_REGEX.findall(summand_no_operaror):
            is_div = False
            factor_no_operator = factor
            if factor[0] == "*":
                factor_no_operator = factor[1:]
            elif factor[0] == "/":
                factor_no_operator = factor[1:]
                is_div = True

            if factor_no_operator.replace(".", "").isnumeric():
                f = float(factor_no_operator)
                coeff *= 1.0 / f if is_div else f
            else:
                var = factor_no_operator

        if var == "":
            expr += coeff
        else:
            is_const = False
            expr += Term(sign * coeff, Variable(var))

    if is_const:
        return expr.constant
    return expr


def _add_operation(
    mqt_computation: QuantumComputation,
    type_: OpType,
    qargs: tuple[Qubit] | list[Qubit],
    params: list[float | ParameterExpression],
    qubit_map: dict[Qubit, int],
) -> None:
    qubits = [qubit_map[qubit] for qubit in qargs]
    target = qubits.pop()
    parameters = [_parse_symbolic_expression(param) for param in params]
    controls = {Control(qubit) for qubit in qubits}
    if all(isinstance(parameter, float) for parameter in parameters):
        float_parameters = [cast(float, parameter) for parameter in parameters]
        mqt_computation.append_operation(
            StandardOperation(mqt_computation.n_qubits, controls, target, type_, float_parameters)
        )
    else:
        mqt_computation.append_operation(
            SymbolicOperation(mqt_computation.n_qubits, controls, target, type_, parameters)
        )
        for parameter in parameters:
            mqt_computation.add_variable(parameter)


def _add_two_target_operation(
    mqt_computation: QuantumComputation,
    type_: OpType,
    qargs: tuple[Qubit] | list[Qubit],
    params: list[float | ParameterExpression],
    qubit_map: dict[Qubit, int],
) -> None:
    qubits = [qubit_map[qubit] for qubit in qargs]
    target1 = qubits.pop()
    target2 = qubits.pop()
    parameters = [_parse_symbolic_expression(param) for param in params]
    controls = {Control(qubit) for qubit in qubits}
    if all(isinstance(parameter, float) for parameter in parameters):
        float_parameters = [cast(float, parameter) for parameter in parameters]
        mqt_computation.append_operation(
            StandardOperation(mqt_computation.n_qubits, controls, target1, target2, type_, float_parameters)
        )
    else:
        mqt_computation.append_operation(
            SymbolicOperation(mqt_computation.n_qubits, controls, target1, target2, type_, parameters)
        )
        for parameter in parameters:
            mqt_computation.add_variable(parameter)


def _get_logical_qubit_indices(mqt_computation: QuantumComputation, layout: Layout) -> dict[Qubit, int]:
    registers = layout.get_registers()
    logical_qubit_index = 0
    logical_qubit_indices = {}
    ancilla_register = None

    for register in registers:
        if register.name == "ancilla":
            ancilla_register = register
            continue

        size = register.size
        for physical_qubit_index in range(size):
            logical_qubit_indices[Qubit(register, physical_qubit_index)] = logical_qubit_index
            logical_qubit_index += 1

    if ancilla_register is not None:
        ancilla_size = ancilla_register.size
        for physical_qubit_index in range(ancilla_size):
            logical_qubit_indices[Qubit(ancilla_register, physical_qubit_index)] = logical_qubit_index
            mqt_computation.set_logical_qubit_ancillary(logical_qubit_index)
            logical_qubit_index += 1

    return logical_qubit_indices


def _import_layouts(mqt_computation: QuantumComputation, qiskit_circuit: QuantumCircuit) -> None:
    initial_layout = qiskit_circuit._layout.initial_layout  # noqa: SLF001
    final_layout = qiskit_circuit._layout.final_layout  # noqa: SLF001

    initial_logical_qubit_indices = _get_logical_qubit_indices(mqt_computation, initial_layout)
    final_logical_qubit_indices = _get_logical_qubit_indices(mqt_computation, final_layout)

    physical_qubits = initial_layout.get_physical_bits()
    for physical_qubit, logical_qubit in physical_qubits.items():
        mqt_computation.initial_layout[physical_qubit] = initial_logical_qubit_indices[logical_qubit]

    physical_qubit = final_layout.get_physical_bits()
    for physical_qubit, logical_qubit in physical_qubits.items():
        mqt_computation.output_permutation[physical_qubit] = final_logical_qubit_indices[logical_qubit]


def _import_definition(
    mqt_computation: QuantumComputation,
    qiskit_circuit: QuantumCircuit,
    qargs: tuple[int] | list[int],
    cargs: tuple[int] | list[int],
    qubit_map: dict[Qubit, int],
    clbit_map: dict[Clbit, int],
) -> None:
    qarg_map = {}
    def_qubits = qiskit_circuit.qubits
    for i in range(len(qargs)):
        qarg_map[def_qubits[i]] = qargs[i]
    carg_map = {}
    def_clbits = qiskit_circuit.clbits
    for i in range(len(cargs)):
        carg_map[def_clbits[i]] = cargs[i]

    data = qiskit_circuit.data
    for inst in data:
        instruction = inst[0]

        mapped_qargs = [qarg_map[qarg] for qarg in inst[1]]
        mapped_cargs = [carg_map[carg] for carg in inst[2]]
        instruction_params = instruction.params
        _emplace_operation(
            mqt_computation, instruction, mapped_qargs, mapped_cargs, instruction_params, qubit_map, clbit_map
        )
