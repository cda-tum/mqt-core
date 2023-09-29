"""Functionality for interoperability with Qiskit."""
from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, cast

from qiskit.circuit import AncillaQubit, AncillaRegister, Clbit, Instruction, ParameterExpression, Qubit

from mqt.core import (
    CompoundOperation,
    Control,
    Expression,
    NonUnitaryOperation,
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

    for inst in qiskit_circuit.data:
        instruction = inst[0]
        qargs = inst[1]
        cargs = inst[2]
        params = instruction.params

        symb_params = _emplace_operation(mqt_computation, instruction, qargs, cargs, params, qubit_map, clbit_map)
        for symb_param in symb_params:
            mqt_computation.add_variable(symb_param)

    # import initial layout and output permutation in case it is available
    if qiskit_circuit.layout is not None:
        _import_layouts(mqt_computation, qiskit_circuit)

    mqt_computation.initialize_io_mapping()
    return mqt_computation


def _emplace_operation(
    mqt_computation: QuantumComputation | CompoundOperation,
    instr: Instruction,
    qargs: list[Qubit] | tuple[Qubit],
    cargs: list[Clbit] | tuple[Clbit],
    params: list[float | ParameterExpression],
    qubit_map: dict[Qubit, int],
    clbit_map: dict[Clbit, int],
) -> list[float | ParameterExpression]:
    name = instr.name

    if name == "measure":
        ctrl = qubit_map[qargs[0]]
        target = clbit_map[cargs[0]]
        mqt_computation.append_operation(NonUnitaryOperation(mqt_computation.n_qubits, [target], [ctrl]))
        return []
    if name == "barrier":
        targets = [qubit_map[qubit] for qubit in qargs]
        mqt_computation.append_operation(NonUnitaryOperation(mqt_computation.n_qubits, targets, OpType.barrier))
        return []
    if name == "reset":
        targets = [qubit_map[qubit] for qubit in qargs]
        mqt_computation.append_operation(NonUnitaryOperation(mqt_computation.n_qubits, targets, OpType.reset))
        return []
    if name in ["i", "id", "iden"]:
        return _add_operation(mqt_computation, OpType.i, qargs, params, qubit_map)
    if name in ["x", "cx", "ccx", "ccx", "mcx", "mcx_gray"]:
        return _add_operation(mqt_computation, OpType.x, qargs, params, qubit_map)
    if name in ["y", "cy"]:
        return _add_operation(mqt_computation, OpType.y, qargs, params, qubit_map)
    if name in ["z", "cz"]:
        return _add_operation(mqt_computation, OpType.z, qargs, params, qubit_map)
    if name in ["h", "ch"]:
        return _add_operation(mqt_computation, OpType.h, qargs, params, qubit_map)
    if name == "s":
        return _add_operation(mqt_computation, OpType.s, qargs, params, qubit_map)
    if name == "sdg":
        return _add_operation(mqt_computation, OpType.sdag, qargs, params, qubit_map)
    if name == "t":
        return _add_operation(mqt_computation, OpType.t, qargs, params, qubit_map)
    if name == "tdg":
        return _add_operation(mqt_computation, OpType.tdag, qargs, params, qubit_map)
    if name in ["rx", "crx", "mcrx"]:
        return _add_operation(mqt_computation, OpType.rx, qargs, params, qubit_map)
    if name in ["ry", "cry", "mcry"]:
        return _add_operation(mqt_computation, OpType.ry, qargs, params, qubit_map)
    if name in ["rz", "crz", "mcrz"]:
        return _add_operation(mqt_computation, OpType.rz, qargs, params, qubit_map)
    if name in ["p", "u1", "cp", "cu1", "mcphase"]:
        return _add_operation(mqt_computation, OpType.phase, qargs, params, qubit_map)
    if name in ["sx", "csx"]:
        return _add_operation(mqt_computation, OpType.sx, qargs, params, qubit_map)
    if name in ["swap", "cswap"]:
        return _add_two_target_operation(mqt_computation, OpType.swap, qargs, params, qubit_map)
    if name == "iswap":
        return _add_two_target_operation(mqt_computation, OpType.iswap, qargs, params, qubit_map)
    if name == "dcx":
        return _add_two_target_operation(mqt_computation, OpType.dcx, qargs, params, qubit_map)
    if name == "ecr":
        return _add_two_target_operation(mqt_computation, OpType.ecr, qargs, params, qubit_map)
    if name == "rxx":
        return _add_two_target_operation(mqt_computation, OpType.rxx, qargs, params, qubit_map)
    if name == "ryy":
        return _add_two_target_operation(mqt_computation, OpType.ryy, qargs, params, qubit_map)
    if name == "rzz":
        return _add_two_target_operation(mqt_computation, OpType.rzz, qargs, params, qubit_map)
    if name == "xx_minus_yy":
        return _add_two_target_operation(mqt_computation, OpType.xx_minus_yy, qargs, params, qubit_map)
    if name == "xx_plus_yy":
        return _add_two_target_operation(mqt_computation, OpType.xx_plus_yy, qargs, params, qubit_map)
    if name == "mcx_recursive":
        if len(qargs) <= 5:
            return _add_operation(mqt_computation, OpType.x, qargs, params, qubit_map)
        return _add_operation(mqt_computation, OpType.x, list(qargs[1:]), params, qubit_map)
    if name == "mcx_vchain":
        size = len(qargs)
        n_controls = (size + 1) // 2
        return _add_operation(mqt_computation, OpType.x, list(qargs[n_controls - 2 :]), params, qubit_map)
    try:
        return _import_definition(mqt_computation, instr.definition, qargs, cargs, qubit_map, clbit_map)
    except Exception:
        print("Failed to import instruction: " + name + " from Qiskit QuantumCircuit\n", sys.stderr)
    return []


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
    mqt_computation: QuantumComputation | CompoundOperation,
    type_: OpType,
    qargs: list[Qubit] | tuple[Qubit],
    params: list[float | ParameterExpression],
    qubit_map: dict[Qubit, int],
) -> list[float | ParameterExpression]:
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
    print(parameters)
    return parameters


def _add_two_target_operation(
    mqt_computation: QuantumComputation | CompoundOperation,
    type_: OpType,
    qargs: list[Qubit] | tuple[Qubit],
    params: list[float | ParameterExpression],
    qubit_map: dict[Qubit, int],
) -> list[float | ParameterExpression]:
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
    return parameters
    # for parameter in parameters:
    #     mqt_computation.add_variable(parameter)


def _get_logical_qubit_indices(mqt_computation: QuantumComputation, layout: Layout) -> dict[Qubit, int]:
    logical_qubit_index = 0
    logical_qubit_indices = {}
    ancilla_register = None

    for register in layout.get_registers():
        if register.name == "ancilla":
            ancilla_register = register
            continue

        for physical_qubit_index in range(register.size):
            logical_qubit_indices[Qubit(register, physical_qubit_index)] = logical_qubit_index
            logical_qubit_index += 1

    if ancilla_register is not None:
        for physical_qubit_index in range(ancilla_register.size):
            logical_qubit_indices[Qubit(ancilla_register, physical_qubit_index)] = logical_qubit_index
            mqt_computation.set_logical_qubit_ancillary(logical_qubit_index)
            logical_qubit_index += 1

    return logical_qubit_indices


def _import_layouts(mqt_computation: QuantumComputation, qiskit_circuit: QuantumCircuit) -> None:
    # qiskit-terra 0.22.0 changed the `_layout` attribute to a
    # `TranspileLayout` dataclass object that contains the initial layout as a
    # `Layout` object in the `initial_layout` attribute.

    initial_layout = qiskit_circuit._layout.initial_layout  # noqa: SLF001
    final_layout = qiskit_circuit._layout.final_layout  # noqa: SLF001

    initial_logical_qubit_indices = _get_logical_qubit_indices(mqt_computation, initial_layout)
    final_logical_qubit_indices = _get_logical_qubit_indices(mqt_computation, final_layout)

    for physical_qubit, logical_qubit in initial_layout.get_physical_bits().items():
        if logical_qubit in initial_logical_qubit_indices:
            mqt_computation.initial_layout[physical_qubit] = initial_logical_qubit_indices[logical_qubit]

    for physical_qubit, logical_qubit in final_layout.get_physical_bits().items():
        if logical_qubit in final_logical_qubit_indices:
            mqt_computation.output_permutation[physical_qubit] = final_logical_qubit_indices[logical_qubit]


def _import_definition(
    mqt_computation: QuantumComputation | CompoundOperation,
    qiskit_circuit: QuantumCircuit,
    qargs: tuple[int] | list[int],
    cargs: tuple[int] | list[int],
    qubit_map: dict[Qubit, int],
    clbit_map: dict[Clbit, int],
) -> list[float | ParameterExpression]:
    qarg_map = {}
    for def_qubit, qarg in zip(qiskit_circuit.qubits, qargs):
        qarg_map[def_qubit] = qarg
    carg_map = {}
    for def_clbit, carg in zip(qiskit_circuit.clbits, cargs):
        carg_map[def_clbit] = carg

    comp_op = CompoundOperation(mqt_computation.n_qubits)

    params = []
    for instruction, qargs, cargs in qiskit_circuit.data:
        mapped_qargs = [qarg_map[qarg] for qarg in qargs]
        mapped_cargs = [carg_map[carg] for carg in cargs]
        instruction_params = instruction.params
        new_params = _emplace_operation(
            comp_op, instruction, mapped_qargs, mapped_cargs, instruction_params, qubit_map, clbit_map
        )
        params.extend(new_params)
    mqt_computation.append_operation(comp_op)
    return params
