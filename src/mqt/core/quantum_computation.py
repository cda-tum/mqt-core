"""Quantum computation core module."""

from __future__ import annotations

from ._core.quantum_computation import QuantumComputation

__all__ = ("QuantumComputation",)

for cls in (QuantumComputation,):
    cls.__module__ = "mqt.core"
del cls
