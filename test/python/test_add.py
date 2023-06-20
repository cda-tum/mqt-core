"""Test the core module."""

from __future__ import annotations

from mqt.core import add


def test_add() -> None:
    """Test the add function."""
    assert add(1, 2) == 3
