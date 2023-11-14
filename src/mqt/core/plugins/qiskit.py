"""Functionality for interoperability with Qiskit."""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, List, cast

from qiskit.circuit import AncillaQubit, AncillaRegister, Clbit, Instruction, ParameterExpression, Qubit

from .. import QuantumComputation
from ..operations import (
    CompoundOperation,
    Control,
    NonUnitaryOperation,
    OpType,
    StandardOperation,
    SymbolicOperation,
)
from ..symbolic import Expression, Term, Variable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from qiskit.circuit import QuantumCircuit


def qiskit_to_mqt(circ: QuantumCircuit) -> QuantumComputation:
    """Convert a Qiskit `QuantumCircuit` to a `QuantumComputation` object.

    Args:
        circ: The Qiskit circuit to convert.

    Returns:
        The converted circuit.
    """
    qc = QuantumComputation()

    if circ.name is not None:
        qc.name = circ.name

    qubit_index = 0
    qubit_map: dict[Qubit, int] = {}
    for reg in circ.qregs:
        size = reg.size
        if isinstance(reg, AncillaRegister):
            qc.add_ancillary_register(size, reg.name)
        else:
            qc.add_qubit_register(size, reg.name)
        for qubit in reg:
            qubit_map[qubit] = qubit_index
            qubit_index += 1

    clbit_index = 0
    clbit_map: dict[Clbit, int] = {}
    for reg in circ.cregs:
        size = reg.size
        qc.add_classical_register(size, reg.name)
        for bit in reg:
            clbit_map[bit] = clbit_index
            clbit_index += 1

    try:
        qc.global_phase = circ.global_phase
    except TypeError:
        warnings.warn(
            "Symbolic global phase values are not supported yet. Setting global phase to 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        qc.global_phase = 0

    for instruction, qargs, cargs in circ.data:
        symb_params = _emplace_operation(qc, instruction, qargs, cargs, instruction.params, qubit_map, clbit_map)
        for symb_param in symb_params:
            qc.add_variable(symb_param)

    # import initial layout and output permutation in case it is available
    if (hasattr(circ, "layout") and circ.layout is not None) or circ._layout is not None:  # noqa: SLF001
        _import_layouts(qc, circ)

    qc.initialize_io_mapping()
    return qc


_NATIVELY_SUPPORTED_GATES = frozenset(
    {
        "i",
        "id",
        "iden",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "p",
        "u1",
        "rx",
        "ry",
        "rz",
        "u2",
        "u",
        "u3",
        "cx",
        "cy",
        "cz",
        "cp",
        "cu1",
        "ch",
        "crx",
        "cry",
        "crz",
        "cu3",
        "ccx",
        "swap",
        "cswap",
        "iswap",
        "sx",
        "sxdg",
        "csx",
        "mcx",
        "mcx_gray",
        "mcx_recursive",
        "mcx_vchain",
        "mcphase",
        "mcrx",
        "mcry",
        "mcrz",
        "dcx",
        "ecr",
        "rxx",
        "ryy",
        "rzx",
        "rzz",
        "xx_minus_yy",
        "xx_plus_yy",
        "reset",
        "barrier",
        "measure",
    }
)


def _emplace_operation(
    qc: QuantumComputation | CompoundOperation,
    instr: Instruction,
    qargs: Sequence[Qubit],
    cargs: Sequence[Clbit],
    params: Sequence[float | ParameterExpression],
    qubit_map: Mapping[Qubit, int],
    clbit_map: Mapping[Clbit, int],
) -> list[float | ParameterExpression]:
    name = instr.name

    if name not in _NATIVELY_SUPPORTED_GATES:
        try:
            return _import_definition(qc, instr.definition, qargs, cargs, qubit_map, clbit_map)
        except Exception as ex:  # PRAGMA: NO COVER
            msg = f"Unsupported gate {name} with definition {instr.definition}"
            raise NotImplementedError(msg) from ex

    qubits = [qubit_map[qubit] for qubit in qargs]

    if name == "measure":
        clbits = [clbit_map[clbit] for clbit in cargs]
        qc.append(NonUnitaryOperation(qc.num_qubits, qubits, clbits))
        return []

    if name == "reset":
        qc.append(NonUnitaryOperation(qc.num_qubits, qubits))
        return []

    if name == "barrier":
        qc.append(StandardOperation(qc.num_qubits, qubits, OpType.barrier))
        return []

    if name in {"i", "id", "iden"}:
        return _add_operation(qc, OpType.i, qargs, params, qubit_map)

    if name in {"x", "cx", "ccx", "mcx", "mcx_gray"}:
        return _add_operation(qc, OpType.x, qargs, params, qubit_map)

    if name in {"y", "cy"}:
        return _add_operation(qc, OpType.y, qargs, params, qubit_map)

    if name in {"z", "cz"}:
        return _add_operation(qc, OpType.z, qargs, params, qubit_map)

    if name in {"h", "ch"}:
        return _add_operation(qc, OpType.h, qargs, params, qubit_map)

    if name == "s":
        return _add_operation(qc, OpType.s, qargs, params, qubit_map)

    if name == "sdg":
        return _add_operation(qc, OpType.sdg, qargs, params, qubit_map)

    if name == "t":
        return _add_operation(qc, OpType.t, qargs, params, qubit_map)

    if name == "tdg":
        return _add_operation(qc, OpType.tdg, qargs, params, qubit_map)

    if name in {"sx", "csx"}:
        return _add_operation(qc, OpType.sx, qargs, params, qubit_map)

    if name == "mcx_recursive":
        if len(qargs) <= 5:
            return _add_operation(qc, OpType.x, qargs, params, qubit_map)
        # reconfigure controls and targets (drops the last qubit as ancilla)
        qargs = qargs[:-1]
        return _add_operation(qc, OpType.x, qargs, params, qubit_map)

    if name == "mcx_vchain":
        size = len(qargs)
        num_controls = (size + 1) // 2
        # reconfigure controls and targets (drops the last num_controls - 2 qubits as ancilla)
        if num_controls > 2:
            qargs = qargs[: -num_controls + 2]
        return _add_operation(qc, OpType.x, qargs, params, qubit_map)

    if name in {"rx", "crx", "mcrx"}:
        return _add_operation(qc, OpType.rx, qargs, params, qubit_map)

    if name in {"ry", "cry", "mcry"}:
        return _add_operation(qc, OpType.ry, qargs, params, qubit_map)

    if name in {"rz", "crz", "mcrz"}:
        return _add_operation(qc, OpType.rz, qargs, params, qubit_map)

    if name in {"p", "u1", "cp", "cu1", "mcphase"}:
        return _add_operation(qc, OpType.p, qargs, params, qubit_map)

    if name == "u2":
        return _add_operation(qc, OpType.u2, qargs, params, qubit_map)

    if name in {"u", "u3", "cu3"}:
        return _add_operation(qc, OpType.u, qargs, params, qubit_map)

    if name in {"swap", "cswap"}:
        return _add_two_target_operation(qc, OpType.swap, qargs, params, qubit_map)

    if name == "iswap":
        return _add_two_target_operation(qc, OpType.iswap, qargs, params, qubit_map)

    if name == "dcx":
        return _add_two_target_operation(qc, OpType.dcx, qargs, params, qubit_map)

    if name == "ecr":
        return _add_two_target_operation(qc, OpType.ecr, qargs, params, qubit_map)

    if name == "rxx":
        return _add_two_target_operation(qc, OpType.rxx, qargs, params, qubit_map)

    if name == "ryy":
        return _add_two_target_operation(qc, OpType.ryy, qargs, params, qubit_map)

    if name == "rzz":
        return _add_two_target_operation(qc, OpType.rzz, qargs, params, qubit_map)

    if name == "rzx":
        return _add_two_target_operation(qc, OpType.rzx, qargs, params, qubit_map)

    if name == "xx_minus_yy":
        return _add_two_target_operation(qc, OpType.xx_minus_yy, qargs, params, qubit_map)

    if name == "xx_plus_yy":
        return _add_two_target_operation(qc, OpType.xx_plus_yy, qargs, params, qubit_map)

    msg = f"Unsupported gate {name}"  # pragma: no cover
    raise NotImplementedError(msg)


_SUM_REGEX = re.compile("[+|-]?[^+-]+")
_PROD_REGEX = re.compile("[*/]?[^*/]+")


def _parse_symbolic_expression(qiskit_expr: ParameterExpression | float) -> float | Expression:
    if isinstance(qiskit_expr, float):
        return qiskit_expr

    expr_str = qiskit_expr.__str__().strip()
    expr = Expression()
    is_const = True
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

            factor_no_operator = factor_no_operator.strip()
            if factor_no_operator.replace(".", "").isnumeric():
                f = float(factor_no_operator)
                coeff *= 1.0 / f if is_div else f
            else:
                var = factor_no_operator

        if not var:
            expr += coeff
        else:
            is_const = False
            expr += Term(Variable(var), sign * coeff)

    if is_const:
        return expr.constant
    return expr


def _add_operation(
    qc: QuantumComputation | CompoundOperation,
    type_: OpType,
    qargs: Sequence[Qubit],
    params: Sequence[float | ParameterExpression],
    qubit_map: Mapping[Qubit, int],
) -> list[float | ParameterExpression]:
    qubits = [qubit_map[qubit] for qubit in qargs]
    target = qubits.pop()
    controls = {Control(qubit) for qubit in qubits}
    parameters = [_parse_symbolic_expression(param) for param in params]
    if any(isinstance(parameter, Expression) for parameter in parameters):
        qc.append(SymbolicOperation(qc.num_qubits, controls, target, type_, parameters))
    else:
        qc.append(StandardOperation(qc.num_qubits, controls, target, type_, cast(List[float], parameters)))
    return parameters


def _add_two_target_operation(
    qc: QuantumComputation | CompoundOperation,
    type_: OpType,
    qargs: Sequence[Qubit],
    params: Sequence[float | ParameterExpression],
    qubit_map: Mapping[Qubit, int],
) -> list[float | ParameterExpression]:
    qubits = [qubit_map[qubit] for qubit in qargs]
    target2 = qubits.pop()
    target1 = qubits.pop()
    controls = {Control(qubit) for qubit in qubits}
    parameters = [_parse_symbolic_expression(param) for param in params]
    if any(isinstance(parameter, Expression) for parameter in parameters):
        qc.append(SymbolicOperation(qc.num_qubits, controls, target1, target2, type_, parameters))
    else:
        qc.append(StandardOperation(qc.num_qubits, controls, target1, target2, type_, cast(List[float], parameters)))
    return parameters


def _import_layouts(qc: QuantumComputation, circ: QuantumCircuit) -> None:
    # qiskit-terra 0.24.0 added a (public) `layout` attribute
    layout = circ.layout if hasattr(circ, "layout") else circ._layout  # noqa: SLF001

    initial_layout = layout.initial_layout

    # The following creates a map of virtual qubits in the layout to an integer index.
    # Using the `get_virtual_bits()` method of the layout guarantees that the order
    # of the qubits is the same as in the original circuit. In contrast, the
    # `get_registers()` method does not guarantee this as it returns a set that
    # reorders the qubits.
    qubit_to_idx: dict[Qubit, int] = {}
    idx = 0
    for qubit in initial_layout.get_virtual_bits():
        qubit_to_idx[qubit] = idx
        if isinstance(qubit, AncillaQubit):
            qc.set_circuit_qubit_ancillary(idx)
        idx += 1

    for register in initial_layout.get_registers():
        is_ancilla = register.name == "ancilla" or isinstance(register, AncillaRegister)
        if not is_ancilla:
            continue
        for qubit in register:
            qc.set_circuit_qubit_ancillary(qubit_to_idx[qubit])

    # create initial layout (and assume identical output permutation)
    for device_qubit, circuit_qubit in initial_layout.get_physical_bits().items():
        idx = qubit_to_idx[circuit_qubit]
        qc.initial_layout[device_qubit] = idx
        qc.output_permutation[device_qubit] = idx

    if not hasattr(layout, "final_layout"):
        return

    final_layout = layout.final_layout
    if final_layout is None:
        return


def _import_definition(
    qc: QuantumComputation | CompoundOperation,
    circ: QuantumCircuit,
    qargs: Sequence[Qubit],
    cargs: Sequence[Clbit],
    qubit_map: Mapping[Qubit, int],
    clbit_map: Mapping[Clbit, int],
) -> list[float | ParameterExpression]:
    qarg_map = dict(zip(circ.qubits, qargs))
    carg_map = dict(zip(circ.clbits, cargs))

    qc.append(CompoundOperation(qc.num_qubits))
    comp_op = cast(CompoundOperation, qc[-1])

    params = []
    for instruction, qargs, cargs in circ.data:
        mapped_qargs = [qarg_map[qarg] for qarg in qargs]
        mapped_cargs = [carg_map[carg] for carg in cargs]
        instruction_params = instruction.params
        new_params = _emplace_operation(
            comp_op, instruction, mapped_qargs, mapped_cargs, instruction_params, qubit_map, clbit_map
        )
        params.extend(new_params)
    return params
