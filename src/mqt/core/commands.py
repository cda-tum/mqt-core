"""Useful commands for obtaining information about mqt-core."""

from __future__ import annotations

from pathlib import Path

DIR = Path(__file__).parent.absolute()


def include_dir() -> Path:
    """Return the path to the mqt-core include directory."""
    return DIR / "include"


def cmake_dir() -> Path:
    """Return the path to the mqt-core CMake module directory."""
    cmake_installed_path = DIR / "share" / "cmake"
    if cmake_installed_path.exists():
        return cmake_installed_path
    msg = "mqt-core not installed, installation required to access the CMake files."
    raise ImportError(msg)
