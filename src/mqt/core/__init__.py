"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

from ._core import OpType, StandardOperation, CompoundOperation, Permutation, QuantumComputation
from ._version import version as __version__

__all__ = ["__version__", "OpType", "StandardOperation", "CompoundOperation", "Permutation", "QuantumComputation"]
