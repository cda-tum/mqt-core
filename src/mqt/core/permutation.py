"""Permutation - A class for representing qubit permutations."""

from __future__ import annotations

from ._core.permutation import Permutation

__all__ = ("Permutation",)

for cls in (Permutation,):
    cls.__module__ = "mqt.core"
del cls
