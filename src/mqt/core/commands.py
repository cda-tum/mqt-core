"""Useful commands for obtaining information about mqt-core."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def include_dir() -> Path:
    """Return the path to the mqt-core include directory."""
    try:
        dist = distribution("mqt-core")
        located_include_dir = Path(dist.locate_file("mqt/core/include/mqt-core"))
        if located_include_dir.exists() and located_include_dir.is_dir():
            return located_include_dir
        msg = "mqt-core include files not found."
        raise FileNotFoundError(msg)
    except PackageNotFoundError:
        msg = "mqt-core not installed, installation required to access the include files."
        raise ImportError(msg) from None


def cmake_dir() -> Path:
    """Return the path to the mqt-core CMake module directory."""
    try:
        dist = distribution("mqt-core")
        located_cmake_dir = Path(dist.locate_file("mqt/core/share/cmake"))
        if located_cmake_dir.exists() and located_cmake_dir.is_dir():
            return located_cmake_dir
        msg = "mqt-core CMake files not found."
        raise FileNotFoundError(msg)
    except PackageNotFoundError:
        msg = "mqt-core not installed, installation required to access the CMake files."
        raise ImportError(msg) from None
