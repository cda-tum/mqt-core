"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

from ._core import (
    CompoundOperation,
    Control,
    OpType,
    Permutation,
    QuantumComputation,
    StandardOperation,
    SymbolicOperation
)

from ._core.symbolic import (
    Expression,
    Term,
    Variable,
)

from ._version import version as __version__

__all__ = [
    "__version__",
    "OpType",
    "StandardOperation",
    "CompoundOperation",
    "Permutation",
    "QuantumComputation",
    "Control",
    "Expression",
    "SymbolicOperation"
]
