from collections.abc import Mapping, Sequence
from typing import overload

from ..._compat.typing import Self
from ..symbolic import Expression, Variable
from .control import Control
from .optype import OpType
from .standard_operation import StandardOperation

class SymbolicOperation(StandardOperation):
    @overload
    def __init__(self: Self) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        target: int,
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        control: Control,
        target: int,
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        control: Control,
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        controls: set[Control],
        target: int,
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        controls: set[Control],
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        controls: set[Control],
        target0: int,
        target1: int,
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    def get_parameter(self: Self, idx: int) -> Expression | float: ...
    def get_parameters(self: Self) -> list[Expression | float]: ...
    def get_instantiated_operation(self: Self, assignment: Mapping[Variable, float]) -> StandardOperation: ...
    def instantiate(self: Self, assignment: Mapping[Variable, float]) -> None: ...
