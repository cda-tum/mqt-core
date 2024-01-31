"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

from ._core import Permutation, QuantumComputation
from ._version import version as __version__
from ._version import version_tuple as version_info
from .commands import cmake_dir, include_dir

__all__ = [
    "Permutation",
    "QuantumComputation",
    "__version__",
    "cmake_dir",
    "include_dir",
    "version_info",
]

for cls in (Permutation, QuantumComputation):
    cls.__module__ = "mqt.core"
del cls
