from typing import Any, Optional, overload
from enum import Enum
import mqt.core

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
    
annotations: _Feature

