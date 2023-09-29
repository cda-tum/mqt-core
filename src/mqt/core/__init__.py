"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

from ._core.operations import (
    Control,
    ControlType,
    Permutation,
    CompoundOperation,
    SymbolicOperation,
    StandardOperation,
    NonUnitaryOperation,
)

from ._core.symbolic import (
    Expression,
    Term,
    Variable,
)

from ._core.quantum_computation import QuantumComputation, OpType

from ._version import version as __version__

__all__ = [
    "__version__",
    "OpType",
    "StandardOperation",
    "CompoundOperation",
    "NonUnitaryOperation",
    "SymbolicOperation",
    "Permutation",
    "QuantumComputation",
    "Control",
    "ControlType",
    "Expression",
    "Term",
    "Variable",
]
