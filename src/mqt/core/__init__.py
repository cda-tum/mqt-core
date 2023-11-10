"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

from . import operations, symbolic
from ._core import Permutation, QuantumComputation
from ._version import version as __version__

__all__ = [
    "operations",
    "Permutation",
    "QuantumComputation",
    "symbolic",
    "__version__",
]

for cls in (Permutation, QuantumComputation):
    cls.__module__ = "mqt.core"
del cls
