"""Test the evaluation of JSON results."""

from __future__ import annotations

from pathlib import Path

import pytest

from mqt.core.evaluation import __flatten_dict, __post_processing, compare

path_base = Path(__file__).resolve().parent / "results_baseline.json"
path_feature = Path(__file__).resolve().parent / "results_feature.json"


def test_flatten_dict() -> None:
    """Test flatten_dict."""
    d1 = {"a": {"b": {"main": 1}}}
    assert __flatten_dict(d1) == {"a.b.main": 1}
    d2 = {"a": {"b": {"main": 1, "447-add-benchmark-suite-in-mqt-core": 2}}, "d": {"main": 2}}
    assert __flatten_dict(d2) == {"a.b.main": 1, "a.b.447-add-benchmark-suite-in-mqt-core": 2, "d.main": 2}


def test_post_processing() -> None:
    """Test postprocessing."""
    with pytest.raises(ValueError, match="Benchmark a.b is missing algorithm, task, number of qubits or metric!"):
        __post_processing("a.b")
    assert __post_processing("a.b.main.feature") == {
        "algorithm": "a",
        "task": "b",
        "num_qubits": "main",
        "component": "",
        "metric": "feature",
    }
    assert __post_processing("a.b.main.feature.algorithm") == {
        "algorithm": "a",
        "task": "b",
        "num_qubits": "main",
        "component": "feature",
        "metric": "algorithm",
    }
    assert __post_processing("RandomClifford.Simulation.14.dd.real_numbers.cache_manager.memory_used_MiB_peak") == {
        "algorithm": "RandomClifford",
        "task": "Simulation",
        "num_qubits": "14",
        "component": "dd_real_numbers_cache_manager",
        "metric": "memory_used_MiB_peak",
    }
    assert __post_processing("QPE.Functionality.15.dd.matrix.unique_table.total.lookups") == {
        "algorithm": "QPE",
        "task": "Functionality",
        "num_qubits": "15",
        "component": "dd_matrix_unique_table",
        "metric": "total_lookups",
    }


def test_compare_with_negative_factor() -> None:
    """Test factor -0.1."""
    with pytest.raises(ValueError, match="Factor must be positive!"):
        compare(path_base, path_feature, factor=-0.1)


def test_compare_with_invalid_sort_option() -> None:
    """Test invalid sort option."""
    with pytest.raises(ValueError, match="Invalid sort option!"):
        compare(path_base, path_feature, sort="after")


def test_compare_with_factor_zero_point_one() -> None:
    """Test factor 0.1."""
    try:
        compare(path_base, path_feature, factor=0.1, only_changed=False, sort="ratio", no_split=False)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_with_factor_zero_point_three() -> None:
    """Test factor 0.3."""
    try:
        compare(path_base, path_feature, factor=0.3, only_changed=False, sort="ratio", no_split=False)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_only_changed() -> None:
    """Test only changed."""
    try:
        compare(path_base, path_feature, factor=0.2, only_changed=True, sort="ratio", no_split=False)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_only_changed_and_no_split() -> None:
    """Test only changed and no split."""
    try:
        compare(path_base, path_feature, factor=0.2, only_changed=True, sort="ratio", no_split=True)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_not_only_changed_and_no_split() -> None:
    """Test only changed and no split."""
    try:
        compare(path_base, path_feature, factor=0.25, only_changed=False, sort="ratio", no_split=True)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_sort_by_experiment() -> None:
    """Test sort by experiment."""
    try:
        compare(path_base, path_feature, factor=0.2, only_changed=True, sort="algorithm", no_split=True)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e
