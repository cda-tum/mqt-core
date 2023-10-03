from typing import ClassVar, overload

from ..._compat.typing import Self

class OpType:
    __members__: ClassVar[dict[OpType, str]]  # readonly
    barrier: ClassVar[OpType]  # value = <OpType.barrier: 3>
    classic_controlled: ClassVar[OpType]  # value = <OpType.classic_controlled: 38>
    compound: ClassVar[OpType]  # value = <OpType.compound: 34>
    dcx: ClassVar[OpType]  # value = <OpType.dcx: 26>
    ecr: ClassVar[OpType]  # value = <OpType.ecr: 27>
    gphase: ClassVar[OpType]  # value = <OpType.gphase: 1>
    h: ClassVar[OpType]  # value = <OpType.h: 4>
    i: ClassVar[OpType]  # value = <OpType.i: 2>
    iswap: ClassVar[OpType]  # value = <OpType.iswap: 23>
    measure: ClassVar[OpType]  # value = <OpType.measure: 35>
    none: ClassVar[OpType]  # value = <OpType.none: 0>
    peres: ClassVar[OpType]  # value = <OpType.peres: 24>
    peresdag: ClassVar[OpType]  # value = <OpType.peresdag: 25>
    phase: ClassVar[OpType]  # value = <OpType.phase: 16>
    reset: ClassVar[OpType]  # value = <OpType.reset: 36>
    rx: ClassVar[OpType]  # value = <OpType.rx: 19>
    rxx: ClassVar[OpType]  # value = <OpType.rxx: 28>
    ry: ClassVar[OpType]  # value = <OpType.ry: 20>
    ryy: ClassVar[OpType]  # value = <OpType.ryy: 29>
    rz: ClassVar[OpType]  # value = <OpType.rz: 21>
    rzx: ClassVar[OpType]  # value = <OpType.rzx: 31>
    rzz: ClassVar[OpType]  # value = <OpType.rzz: 30>
    s: ClassVar[OpType]  # value = <OpType.s: 8>
    sdag: ClassVar[OpType]  # value = <OpType.sdag: 9>
    swap: ClassVar[OpType]  # value = <OpType.swap: 22>
    sx: ClassVar[OpType]  # value = <OpType.sx: 17>
    sxdag: ClassVar[OpType]  # value = <OpType.sxdag: 18>
    t: ClassVar[OpType]  # value = <OpType.t: 10>
    tdag: ClassVar[OpType]  # value = <OpType.tdag: 11>
    teleportation: ClassVar[OpType]  # value = <OpType.teleportation: 37>
    u2: ClassVar[OpType]  # value = <OpType.u2: 15>
    u3: ClassVar[OpType]  # value = <OpType.u3: 14>
    v: ClassVar[OpType]  # value = <OpType.v: 12>
    vdag: ClassVar[OpType]  # value = <OpType.vdag: 13>
    x: ClassVar[OpType]  # value = <OpType.x: 5>
    xx_minus_yy: ClassVar[OpType]  # value = <OpType.xx_minus_yy: 32>
    xx_plus_yy: ClassVar[OpType]  # value = <OpType.xx_plus_yy: 33>
    y: ClassVar[OpType]  # value = <OpType.y: 6>
    z: ClassVar[OpType]  # value = <OpType.z: 7>
    @property
    def name(self: Self) -> str: ...
    @property
    def value(self: Self) -> int: ...
    def __eq__(self: Self, other: object) -> bool: ...
    def __getstate__(self: Self) -> int: ...
    def __hash__(self: Self) -> int: ...
    def __index__(self: Self) -> int: ...
    @overload
    def __init__(self: Self, value: int) -> None: ...
    @overload
    def __init__(self: Self, arg0: str) -> None: ...
    def __int__(self: Self) -> int: ...
    def __ne__(self: Self, other: object) -> bool: ...
    def __setstate__(self: Self, state: int) -> None: ...
