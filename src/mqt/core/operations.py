"""Operations Module."""

from __future__ import annotations

from ._core.operations import (
    ClassicControlledOperation,
    CompoundOperation,
    Control,
    NonUnitaryOperation,
    Operation,
    OpType,
    StandardOperation,
    SymbolicOperation,
)

__all__ = (
    "ClassicControlledOperation",
    "CompoundOperation",
    "Control",
    "NonUnitaryOperation",
    "Operation",
    "OpType",
    "StandardOperation",
    "SymbolicOperation",
)

for cls in (
    ClassicControlledOperation,
    CompoundOperation,
    Control,
    NonUnitaryOperation,
    Operation,
    OpType,
    StandardOperation,
    SymbolicOperation,
):
    cls.__module__ = "mqt.core.operations"
del cls
