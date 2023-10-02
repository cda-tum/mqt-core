"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

from ._core.operations import (
    CompoundOperation,
    Control,
    ControlType,
    NonUnitaryOperation,
    Permutation,
    StandardOperation,
    SymbolicOperation,
)
from ._core.quantum_computation import OpType, QuantumComputation
from ._core.symbolic import (
    Expression,
    Term,
    Variable,
)
from ._version import version as __version__

__all__ = [
    "__version__",
    "CompoundOperation",
    "Control",
    "ControlType",
    "NonUnitaryOperation",
    "Permutation",
    "StandardOperation",
    "SymbolicOperation",
    "OpType",
    "QuantumComputation",
    "Expression",
    "Term",
    "Variable",
]
