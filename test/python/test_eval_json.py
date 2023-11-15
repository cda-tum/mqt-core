"""Test the evaluation of JSON results."""

from __future__ import annotations

from pathlib import Path

import pytest

from mqt.core.evaluation import compare, flatten_dict, higher_better, is_nested

path = Path(__file__).resolve().parent / "example_results.json"


def test_higher_better() -> None:
    """Test if a metric is better if higher."""
    assert higher_better("BV_Functionality_1024_dd_matrix_unique_table_total_hits")
    assert higher_better("BV_Functionality_1024_dd_compute_tables_matrix_add_hit_ratio")
    assert not higher_better("GHZ_Simulation_256_dd_matrix_unique_table_total_num_active_entries")
    assert not higher_better("WState_Simulation_1024_dd_real_numbers_cache_manager_num_available_for_reuse_peak")


def test_is_nested() -> None:
    """Test is_nested."""
    assert is_nested({"a": {"b": 1}})
    assert not is_nested({"a": 1})
    assert not is_nested({"a": {"b": {"c": 1}}})


def test_flatten_dict() -> None:
    """Test flatten_dict."""
    d1 = {"a": {"b": 1}}
    assert flatten_dict(d1) == {"a_b": [1, "skipped"]}
    d2 = {"a": {"b": {"c": 1}}}
    assert flatten_dict(d2) == {"a_b_c": [1, "skipped"]}
    d3 = {"a": {"b": {"c": 1}}, "d": {"e": 2}}
    assert flatten_dict(d3) == {"a_b_c": [1, "skipped"], "d_e": [2, "skipped"]}


def test_compare_with_negative_factor() -> None:
    """Test factor -0.1."""
    with pytest.raises(ValueError, match="Factor must be positive!"):
        compare(path, factor=-0.1)


def test_compare_with_invalid_sort_option() -> None:
    """Test invalid sort option."""
    with pytest.raises(ValueError, match="Invalid sort option!"):
        compare(path, sort="after")


def test_compare_with_factor_zero_point_one() -> None:
    """Test factor 0.1."""
    try:
        compare(path, factor=0.1, only_changed=False, sort="ratio", no_split=False)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_with_factor_zero_point_three() -> None:
    """Test factor 0.3."""
    try:
        compare(path, factor=0.3, only_changed=False, sort="ratio", no_split=False)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_only_changed() -> None:
    """Test only changed."""
    try:
        compare(path, factor=0.2, only_changed=True, sort="ratio", no_split=False)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_only_changed_and_no_split() -> None:
    """Test only changed and no split."""
    try:
        compare(path, factor=0.2, only_changed=True, sort="ratio", no_split=True)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_not_only_changed_and_no_split() -> None:
    """Test only changed and no split."""
    try:
        compare(path, factor=0.2, only_changed=False, sort="ratio", no_split=True)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_sort_by_experiment() -> None:
    """Test sort by experiment."""
    try:
        compare(path, factor=0.2, only_changed=True, sort="experiment", no_split=True)
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e
