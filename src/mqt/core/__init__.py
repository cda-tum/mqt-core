"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

import sys

# under Windows, make sure to add the appropriate DLL directory to the PATH
if sys.platform == "win32":

    def _dll_patch() -> None:
        """Add the DLL directory to the PATH."""
        import os
        import sysconfig
        from pathlib import Path

        bin_dir = Path(sysconfig.get_paths()["purelib"]) / "mqt" / "core" / "bin"
        os.add_dll_directory(str(bin_dir))

    _dll_patch()
    del _dll_patch

from ._core import Permutation, QuantumComputation
from ._version import version as __version__
from ._version import version_tuple as version_info

__all__ = [
    "Permutation",
    "QuantumComputation",
    "__version__",
    "version_info",
]

for cls in (Permutation, QuantumComputation):
    cls.__module__ = "mqt.core"
del cls
