from collections.abc import Sequence
from typing import overload

from ..._compat.typing import Self
from .control import Control
from .operation import Operation
from .optype import OpType

class StandardOperation(Operation):
    @overload
    def __init__(self: Self) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        target: int,
        op_type: OpType,
        params: Sequence[float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        control: Control,
        target: int,
        op_type: OpType,
        params: Sequence[float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        control: Control,
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        controls: set[Control],
        target: int,
        op_type: OpType,
        params: Sequence[float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        controls: set[Control],
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    @overload
    def __init__(self: Self, nq: int, controls: set[Control], target: int, starting_qubit: int = 0) -> None: ...
    @overload
    def __init__(
        self: Self,
        nq: int,
        controls: set[Control],
        target0: int,
        target1: int,
        op_type: OpType,
        params: Sequence[float] | None = None,
        starting_qubit: int = 0,
    ) -> None: ...
    def add_control(self: Self, control: Control) -> None: ...
    def clear_controls(self: Self) -> None: ...
    def remove_control(self: Self, control: Control) -> None: ...
    def invert(self: Self) -> None: ...
    def qasm_str(self: Self, qreg: Sequence[tuple[str, str]], creg: Sequence[tuple[str, str]]) -> str: ...
