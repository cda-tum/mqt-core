"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

from ._core import (
    ClassicControlledOperation,
    CompoundOperation,
    Control,
    NonUnitaryOperation,
    Operation,
    OpType,
    StandardOperation,
    SymbolicOperation,
)
from ._core import Expression, Term, Variable
from ._core import Permutation, QuantumComputation
from ._version import version as __version__

__all__ = [
    "operations",
    "Permutation",
    "QuantumComputation",
    "symbolic",
    "__version__",
    "Variable", "Term", "Expression",
        "ClassicControlledOperation",
    "CompoundOperation",
    "Control",
    "NonUnitaryOperation",
    "Operation",
    "OpType",
    "StandardOperation",
    "SymbolicOperation",
]

for cls in (Permutation, QuantumComputation):
    cls.__module__ = "mqt.core"
del cls
