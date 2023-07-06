from typing import Any, Optional, overload
from enum import Enum
import mqt.core._core

class CompoundOperation:

    def __init__(self, arg0: int, arg1: list[mqt.core._core.Operation], /) -> None:
        ...

    def acts_on(self, arg: int, /) -> bool:
        ...

    def add_depth_contribution(self, arg: list[int], /) -> None:
        ...

    def clone(self) -> mqt.core._core.Operation:
        ...

    @property
    def (self) -> set[mqt.core._core.Control]:
        ...
    @controls.setter
    def (self) -> set[mqt.core._core.Control]:
        ...

    def empty(self) -> bool:
        ...

    def equals(self, arg0: mqt.core._core.Operation, arg1: mqt.core._core.Permutation, arg2: mqt.core._core.Permutation, /) -> bool:
        ...

    @property
    def (self, arg: mqt.core._core.OpType, /) -> None:
        ...
    @gate.setter
    def (self, arg: mqt.core._core.OpType, /) -> None:
        ...

    def get_starting_qubit(self) -> int:
        ...

    def get_used_qubits(self) -> set[int]:
        ...

    def is_classic_controlled_operation(self) -> bool:
        ...

    def is_compound_operation(self) -> bool:
        ...

    def is_controlled(self) -> bool:
        ...

    def is_non_unitary_operation(self) -> bool:
        ...

    def is_standard_operation(self) -> bool:
        ...

    def is_symbolic_operation(self) -> bool:
        ...

    def is_unitary(self) -> bool:
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...
    @n_qubits.setter
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> str:
        ...
    @name.setter
    def (self) -> str:
        ...

    def set_n_qubits(self, arg: int, /) -> None:
        ...

    def size(self) -> int:
        ...

    @property
    def (self) -> list[int]:
        ...
    @targets.setter
    def (self) -> list[int]:
        ...

    @property
    def (self) -> mqt.core._core.OpType:
        ...

class Control:

    def __init__(self, arg0: int, arg1: mqt.core._core.ControlType, /) -> None:
        """
        __init__(self, arg0: int, arg1: mqt.core._core.ControlType, /) -> None
        """
        ...

    @overload
    def __init__(self, arg: int, /) -> None:
        """
        __init__(self, arg: int, /) -> None
        """
        ...

    @property
    def (self) -> int:
        ...
    @qubit.setter
    def (self) -> int:
        ...

    @property
    def (self) -> mqt.core._core.ControlType:
        ...
    @type.setter
    def (self) -> mqt.core._core.ControlType:
        ...

class ControlType:
    """
    <attribute '__doc__' of 'ControlType' objects>
    """

    @entries: dict

    Neg: ControlType

    Pos: ControlType

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

Neg: ControlType

class OpType:
    """
    <attribute '__doc__' of 'OpType' objects>
    """

    @entries: dict

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    barrier: OpType

    classiccontrolled: OpType

    compound: OpType

    dcx: OpType

    ecr: OpType

    def from_string(arg: str, /) -> mqt.core._core.OpType:
        ...

    gphase: OpType

    h: OpType

    i: OpType

    iswap: OpType

    measure: OpType

    none: OpType

    peres: OpType

    peresdag: OpType

    phase: OpType

    reset: OpType

    rx: OpType

    rxx: OpType

    ry: OpType

    ryy: OpType

    rz: OpType

    rzx: OpType

    rzz: OpType

    s: OpType

    sdag: OpType

    showprobabilities: OpType

    snapshot: OpType

    swap: OpType

    sx: OpType

    sxdag: OpType

    t: OpType

    tdag: OpType

    teleportation: OpType

    u2: OpType

    u3: OpType

    v: OpType

    vdag: OpType

    x: OpType

    xx_minus_yy: OpType

    xx_plus_yy: OpType

    y: OpType

    z: OpType

class Operation:

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    def acts_on(self, arg: int, /) -> bool:
        ...

    @property
    def (self) -> set[mqt.core._core.Control]:
        ...
    @controls.setter
    def (self) -> set[mqt.core._core.Control]:
        ...

    @property
    def (self, arg: mqt.core._core.OpType, /) -> None:
        ...
    @gate.setter
    def (self, arg: mqt.core._core.OpType, /) -> None:
        ...

    def get_starting_qubit(self) -> int:
        ...

    def get_used_qubits(self) -> set[int]:
        ...

    def is_classic_controlled_operation(self) -> bool:
        ...

    def is_compound_operation(self) -> bool:
        ...

    def is_controlled(self) -> bool:
        ...

    def is_non_unitary_operation(self) -> bool:
        ...

    def is_standard_operation(self) -> bool:
        ...

    def is_symbolic_operation(self) -> bool:
        ...

    def is_unitary(self) -> bool:
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...
    @n_qubits.setter
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> str:
        ...
    @name.setter
    def (self) -> str:
        ...

    @property
    def (self) -> list[int]:
        ...
    @targets.setter
    def (self) -> list[int]:
        ...

    @property
    def (self) -> mqt.core._core.OpType:
        ...

class Permutation:

    def __init__(self, arg: dict[int, int], /) -> None:
        ...

    def apply(self, arg: list[int], /) -> list[int]:
        """
        apply(self, arg: list[int], /) -> list[int]
        """
        ...

    @overload
    def apply(self, arg: set[mqt.core._core.Control], /) -> set[mqt.core._core.Control]:
        """
        apply(self, arg: set[mqt.core._core.Control], /) -> set[mqt.core._core.Control]
        """
        ...

Pos: ControlType

class QuantumComputation:
    """
    Representation of quantum circuits within MQT Core
    """

    def __init__(self, filename: str) -> None:
        """
        Read QuantumComputation from given file. Supported formats are [OpenQASM, Real, GRCS, TFC, QC]
        """
        ...

    @overload
    def __init__(self, nq: int) -> None:
        """
        Constructs an empty QuantumComputation with the given number of qubits.
        """
        ...

    def barrier(self, arg: list[int], /) -> None:
        """
        barrier(self, arg: list[int], /) -> None
        """
        ...

    @overload
    def barrier(self, arg: int, /) -> None:
        """
        barrier(self, arg: int, /) -> None
        """
        ...

    def classic_controlled(self, arg0: mqt.core._core.OpType, arg1: int, arg2: set[mqt.core._core.Control], arg3: tuple[int, int], arg4: int, arg5: list[float], /) -> None:
        """
        classic_controlled(self, arg0: mqt.core._core.OpType, arg1: int, arg2: set[mqt.core._core.Control], arg3: tuple[int, int], arg4: int, arg5: list[float], /) -> None
        """
        ...

    @overload
    def classic_controlled(self, arg0: mqt.core._core.OpType, arg1: int, arg2: tuple[int, int], arg3: int, arg4: list[float], /) -> None:
        """
        classic_controlled(self, arg0: mqt.core._core.OpType, arg1: int, arg2: tuple[int, int], arg3: int, arg4: list[float], /) -> None
        """
        ...

    @overload
    def classic_controlled(self, arg0: mqt.core._core.OpType, arg1: int, arg2: mqt.core._core.Control, arg3: tuple[int, int], arg4: int, arg5: list[float], /) -> None:
        """
        classic_controlled(self, arg0: mqt.core._core.OpType, arg1: int, arg2: mqt.core._core.Control, arg3: tuple[int, int], arg4: int, arg5: list[float], /) -> None
        """
        ...

    def clone(self) -> mqt.core._core.QuantumComputation:
        """
        Clone this QuantumComputation object.
        """
        ...

    def dcx(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None:
        """
        dcx(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def dcx(self, arg0: int, arg1: int, /) -> None:
        """
        dcx(self, arg0: int, arg1: int, /) -> None
        """
        ...

    @overload
    def dcx(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None:
        """
        dcx(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None
        """
        ...

    @property
    def (self) -> int:
        ...

    def ecr(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None:
        """
        ecr(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def ecr(self, arg0: int, arg1: int, /) -> None:
        """
        ecr(self, arg0: int, arg1: int, /) -> None
        """
        ...

    @overload
    def ecr(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None:
        """
        ecr(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None
        """
        ...

    @property
    def (self) -> float:
        ...
    @gphase.setter
    def (self) -> float:
        ...

    def h(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        h(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def h(self, arg: int, /) -> None:
        """
        h(self, arg: int, /) -> None
        """
        ...

    @overload
    def h(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        h(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def i(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        i(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def i(self, arg: int, /) -> None:
        """
        i(self, arg: int, /) -> None
        """
        ...

    @overload
    def i(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        i(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def iswap(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None:
        """
        iswap(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def iswap(self, arg0: int, arg1: int, /) -> None:
        """
        iswap(self, arg0: int, arg1: int, /) -> None
        """
        ...

    @overload
    def iswap(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None:
        """
        iswap(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None
        """
        ...

    def measure(self, arg0: list[int], arg1: list[int], /) -> None:
        """
        measure(self, arg0: list[int], arg1: list[int], /) -> None
        """
        ...

    @overload
    def measure(self, arg0: int, arg1: int, /) -> None:
        """
        measure(self, arg0: int, arg1: int, /) -> None
        """
        ...

    @overload
    def measure(self, arg0: int, arg1: tuple[str, int], /) -> None:
        """
        measure(self, arg0: int, arg1: tuple[str, int], /) -> None
        """
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...

    def peres(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None:
        """
        peres(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def peres(self, arg0: int, arg1: int, /) -> None:
        """
        peres(self, arg0: int, arg1: int, /) -> None
        """
        ...

    @overload
    def peres(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None:
        """
        peres(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None
        """
        ...

    def peresdag(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None:
        """
        peresdag(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def peresdag(self, arg0: int, arg1: int, /) -> None:
        """
        peresdag(self, arg0: int, arg1: int, /) -> None
        """
        ...

    @overload
    def peresdag(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None:
        """
        peresdag(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None
        """
        ...

    def phase(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, /) -> None:
        """
        phase(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, /) -> None
        """
        ...

    @overload
    def phase(self, arg0: int, arg1: float, /) -> None:
        """
        phase(self, arg0: int, arg1: float, /) -> None
        """
        ...

    @overload
    def phase(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, /) -> None:
        """
        phase(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, /) -> None
        """
        ...

    def reset(self, arg: int, /) -> None:
        """
        reset(self, arg: int, /) -> None
        """
        ...

    @overload
    def reset(self, arg: int, /) -> None:
        """
        reset(self, arg: int, /) -> None
        """
        ...

    @overload
    def reset(self, arg: list[int], /) -> None:
        """
        reset(self, arg: list[int], /) -> None
        """
        ...

    def rx(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, /) -> None:
        """
        rx(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, /) -> None
        """
        ...

    @overload
    def rx(self, arg0: int, arg1: float, /) -> None:
        """
        rx(self, arg0: int, arg1: float, /) -> None
        """
        ...

    @overload
    def rx(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, /) -> None:
        """
        rx(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, /) -> None
        """
        ...

    def rxx(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, /) -> None:
        """
        rxx(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, /) -> None
        """
        ...

    @overload
    def rxx(self, arg0: int, arg1: int, arg2: float, /) -> None:
        """
        rxx(self, arg0: int, arg1: int, arg2: float, /) -> None
        """
        ...

    @overload
    def rxx(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, /) -> None:
        """
        rxx(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, /) -> None
        """
        ...

    def ry(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, /) -> None:
        """
        ry(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, /) -> None
        """
        ...

    @overload
    def ry(self, arg0: int, arg1: float, /) -> None:
        """
        ry(self, arg0: int, arg1: float, /) -> None
        """
        ...

    @overload
    def ry(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, /) -> None:
        """
        ry(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, /) -> None
        """
        ...

    def ryy(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, /) -> None:
        """
        ryy(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, /) -> None
        """
        ...

    @overload
    def ryy(self, arg0: int, arg1: int, arg2: float, /) -> None:
        """
        ryy(self, arg0: int, arg1: int, arg2: float, /) -> None
        """
        ...

    @overload
    def ryy(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, /) -> None:
        """
        ryy(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, /) -> None
        """
        ...

    def rz(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, /) -> None:
        """
        rz(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, /) -> None
        """
        ...

    @overload
    def rz(self, arg0: int, arg1: float, /) -> None:
        """
        rz(self, arg0: int, arg1: float, /) -> None
        """
        ...

    @overload
    def rz(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, /) -> None:
        """
        rz(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, /) -> None
        """
        ...

    def rzx(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, /) -> None:
        """
        rzx(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, /) -> None
        """
        ...

    @overload
    def rzx(self, arg0: int, arg1: int, arg2: float, /) -> None:
        """
        rzx(self, arg0: int, arg1: int, arg2: float, /) -> None
        """
        ...

    @overload
    def rzx(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, /) -> None:
        """
        rzx(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, /) -> None
        """
        ...

    def rzz(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, /) -> None:
        """
        rzz(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, /) -> None
        """
        ...

    @overload
    def rzz(self, arg0: int, arg1: int, arg2: float, /) -> None:
        """
        rzz(self, arg0: int, arg1: int, arg2: float, /) -> None
        """
        ...

    @overload
    def rzz(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, /) -> None:
        """
        rzz(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, /) -> None
        """
        ...

    def s(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        s(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def s(self, arg: int, /) -> None:
        """
        s(self, arg: int, /) -> None
        """
        ...

    @overload
    def s(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        s(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def sdag(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        sdag(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def sdag(self, arg: int, /) -> None:
        """
        sdag(self, arg: int, /) -> None
        """
        ...

    @overload
    def sdag(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        sdag(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def swap(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None:
        """
        swap(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def swap(self, arg0: int, arg1: int, /) -> None:
        """
        swap(self, arg0: int, arg1: int, /) -> None
        """
        ...

    @overload
    def swap(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None:
        """
        swap(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, /) -> None
        """
        ...

    def sx(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        sx(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def sx(self, arg: int, /) -> None:
        """
        sx(self, arg: int, /) -> None
        """
        ...

    @overload
    def sx(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        sx(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def sxdag(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        sxdag(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def sxdag(self, arg: int, /) -> None:
        """
        sxdag(self, arg: int, /) -> None
        """
        ...

    @overload
    def sxdag(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        sxdag(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def t(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        t(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def t(self, arg: int, /) -> None:
        """
        t(self, arg: int, /) -> None
        """
        ...

    @overload
    def t(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        t(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def tdag(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        tdag(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def tdag(self, arg: int, /) -> None:
        """
        tdag(self, arg: int, /) -> None
        """
        ...

    @overload
    def tdag(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        tdag(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def u2(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, arg3: float, /) -> None:
        """
        u2(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, arg3: float, /) -> None
        """
        ...

    @overload
    def u2(self, arg0: int, arg1: float, arg2: float, /) -> None:
        """
        u2(self, arg0: int, arg1: float, arg2: float, /) -> None
        """
        ...

    @overload
    def u2(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, arg3: float, /) -> None:
        """
        u2(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, arg3: float, /) -> None
        """
        ...

    def u3(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, arg3: float, arg4: float, /) -> None:
        """
        u3(self, arg0: int, arg1: set[mqt.core._core.Control], arg2: float, arg3: float, arg4: float, /) -> None
        """
        ...

    @overload
    def u3(self, arg0: int, arg1: float, arg2: float, arg3: float, /) -> None:
        """
        u3(self, arg0: int, arg1: float, arg2: float, arg3: float, /) -> None
        """
        ...

    @overload
    def u3(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, arg3: float, arg4: float, /) -> None:
        """
        u3(self, arg0: int, arg1: mqt.core._core.Control, arg2: float, arg3: float, arg4: float, /) -> None
        """
        ...

    def v(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        v(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def v(self, arg: int, /) -> None:
        """
        v(self, arg: int, /) -> None
        """
        ...

    @overload
    def v(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        v(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def vdag(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        vdag(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def vdag(self, arg: int, /) -> None:
        """
        vdag(self, arg: int, /) -> None
        """
        ...

    @overload
    def vdag(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        vdag(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def x(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        x(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def x(self, arg: int, /) -> None:
        """
        x(self, arg: int, /) -> None
        """
        ...

    @overload
    def x(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        x(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def xx_minus_yy(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, arg4: float, /) -> None:
        """
        xx_minus_yy(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, arg4: float, /) -> None
        """
        ...

    @overload
    def xx_minus_yy(self, arg0: int, arg1: int, arg2: float, arg3: float, /) -> None:
        """
        xx_minus_yy(self, arg0: int, arg1: int, arg2: float, arg3: float, /) -> None
        """
        ...

    @overload
    def xx_minus_yy(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, arg4: float, /) -> None:
        """
        xx_minus_yy(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, arg4: float, /) -> None
        """
        ...

    def xx_plus_yy(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, arg4: float, /) -> None:
        """
        xx_plus_yy(self, arg0: int, arg1: int, arg2: set[mqt.core._core.Control], arg3: float, arg4: float, /) -> None
        """
        ...

    @overload
    def xx_plus_yy(self, arg0: int, arg1: int, arg2: float, arg3: float, /) -> None:
        """
        xx_plus_yy(self, arg0: int, arg1: int, arg2: float, arg3: float, /) -> None
        """
        ...

    @overload
    def xx_plus_yy(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, arg4: float, /) -> None:
        """
        xx_plus_yy(self, arg0: int, arg1: int, arg2: mqt.core._core.Control, arg3: float, arg4: float, /) -> None
        """
        ...

    def y(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        y(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def y(self, arg: int, /) -> None:
        """
        y(self, arg: int, /) -> None
        """
        ...

    @overload
    def y(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        y(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

    def z(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None:
        """
        z(self, arg0: int, arg1: set[mqt.core._core.Control], /) -> None
        """
        ...

    @overload
    def z(self, arg: int, /) -> None:
        """
        z(self, arg: int, /) -> None
        """
        ...

    @overload
    def z(self, arg0: int, arg1: mqt.core._core.Control, /) -> None:
        """
        z(self, arg0: int, arg1: mqt.core._core.Control, /) -> None
        """
        ...

class StandardOperation:

    def __init__(self, nq: int, controls: set[mqt.core._core.Control], target0: int, target1: int, op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None:
        """
        __init__(self, nq: int, controls: set[mqt.core._core.Control], target0: int, target1: int, op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None
        """
        ...

    @overload
    def __init__(self) -> None:
        """
        __init__(self) -> None
        """
        ...

    @overload
    def __init__(self, nq: int, target: int, op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None:
        """
        __init__(self, nq: int, target: int, op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None
        """
        ...

    @overload
    def __init__(self, nq: int, targets: list[int], op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None:
        """
        __init__(self, nq: int, targets: list[int], op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None
        """
        ...

    @overload
    def __init__(self, nq: int, control: mqt.core._core.Control, target: int, op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None:
        """
        __init__(self, nq: int, control: mqt.core._core.Control, target: int, op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None
        """
        ...

    @overload
    def __init__(self, nq: int, control: mqt.core._core.Control, targets: list[int], op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None:
        """
        __init__(self, nq: int, control: mqt.core._core.Control, targets: list[int], op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None
        """
        ...

    @overload
    def __init__(self, nq: int, controls: set[mqt.core._core.Control], target: int, op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None:
        """
        __init__(self, nq: int, controls: set[mqt.core._core.Control], target: int, op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None
        """
        ...

    @overload
    def __init__(self, nq: int, controls: set[mqt.core._core.Control], targets: list[int], op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None:
        """
        __init__(self, nq: int, controls: set[mqt.core._core.Control], targets: list[int], op_type: mqt.core._core.OpType, params: list[float] = [], starting_qubit: int = 0) -> None
        """
        ...

    @overload
    def __init__(self, nq: int, controls: set[mqt.core._core.Control], target: int, starting_qubit: int = 0) -> None:
        """
        __init__(self, nq: int, controls: set[mqt.core._core.Control], target: int, starting_qubit: int = 0) -> None
        """
        ...

    def acts_on(self, arg: int, /) -> bool:
        ...

    def clone(self) -> mqt.core._core.Operation:
        ...

    @property
    def (self) -> set[mqt.core._core.Control]:
        ...
    @controls.setter
    def (self) -> set[mqt.core._core.Control]:
        ...

    def equals(self, arg0: mqt.core._core.Operation, arg1: mqt.core._core.Permutation, arg2: mqt.core._core.Permutation, /) -> bool:
        """
        equals(self, arg0: mqt.core._core.Operation, arg1: mqt.core._core.Permutation, arg2: mqt.core._core.Permutation, /) -> bool
        """
        ...

    @overload
    def equals(self, arg: mqt.core._core.Operation, /) -> bool:
        """
        equals(self, arg: mqt.core._core.Operation, /) -> bool
        """
        ...

    @property
    def (self, arg: mqt.core._core.OpType, /) -> None:
        ...
    @gate.setter
    def (self, arg: mqt.core._core.OpType, /) -> None:
        ...

    def get_starting_qubit(self) -> int:
        ...

    def get_used_qubits(self) -> set[int]:
        ...

    def is_classic_controlled_operation(self) -> bool:
        ...

    def is_compound_operation(self) -> bool:
        ...

    def is_controlled(self) -> bool:
        ...

    def is_non_unitary_operation(self) -> bool:
        ...

    def is_standard_operation(self) -> bool:
        ...

    def is_symbolic_operation(self) -> bool:
        ...

    def is_unitary(self) -> bool:
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...
    @n_qubits.setter
    def (self) -> int:
        ...

    @property
    def (self) -> int:
        ...

    @property
    def (self) -> str:
        ...
    @name.setter
    def (self) -> str:
        ...

    @property
    def (self) -> list[int]:
        ...
    @targets.setter
    def (self) -> list[int]:
        ...

    @property
    def (self) -> mqt.core._core.OpType:
        ...

barrier: OpType

classiccontrolled: OpType

compound: OpType

dcx: OpType

ecr: OpType

gphase: OpType

h: OpType

i: OpType

iswap: OpType

measure: OpType

none: OpType

peres: OpType

peresdag: OpType

phase: OpType

reset: OpType

rx: OpType

rxx: OpType

ry: OpType

ryy: OpType

rz: OpType

rzx: OpType

rzz: OpType

s: OpType

sdag: OpType

showprobabilities: OpType

snapshot: OpType

swap: OpType

sx: OpType

sxdag: OpType

t: OpType

tdag: OpType

teleportation: OpType

u2: OpType

u3: OpType

v: OpType

vdag: OpType

x: OpType

xx_minus_yy: OpType

xx_plus_yy: OpType

y: OpType

z: OpType
