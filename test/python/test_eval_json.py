"""Test the evaluation of JSON results."""

from __future__ import annotations

from pathlib import Path
from mqt.core.evaluation import compare

path = Path(__file__).resolve().parent / "example_results.json"




def test_zero_point_one() -> None:
    """Test factor 0.1."""
    compare(path, factor=0.1, only_changed=False, sort="ratio", no_split=False)


def test_zero_point_three() -> None:
    """Test factor 0.3."""
    compare(path, factor=0.3, only_changed=False, sort="ratio", no_split=False)


def test_only_changed() -> None:
    """Test only changed."""
    compare(path, factor=0.2, only_changed=True, sort="ratio", no_split=False)


def test_only_changed_and_no_split() -> None:
    """Test only changed and no split."""
    compare(path, factor=0.2, only_changed=True, sort="ratio", no_split=True)


def test_sort_by_experiment() -> None:
    """Test sort by experiment."""
    compare(path, factor=0.2, only_changed=True, sort="experiment", no_split=True)
