"""Test the evaluation of JSON results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mqt.core.evaluation import __aggregate, __flatten_dict, __post_processing, compare

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


def test_aggregate() -> None:
    """Test the data aggregation method."""
    try:
        df_all = __aggregate(path_base, path_feature)
        lookups = df_all[df_all["metric"] == "lookups"]
        assert len(lookups.index) == 1
        assert lookups["before"].values[0] == 8172
        assert lookups["after"].values[0] == 0
        assert lookups["ratio"].values[0] == 0.0
        hit_ratio = df_all[df_all["metric"] == "hit_ratio"]
        assert len(hit_ratio.index) == 1
        assert hit_ratio["before"].values[0] == 0.5
        assert hit_ratio["after"].values[0] == 0.8
        assert hit_ratio["ratio"].values[0] == 1.6
        memory_mib = df_all[df_all["metric"] == "memory_MiB"]
        assert len(memory_mib.index) == 1
        assert memory_mib["after"].values[0] == "skipped"
        assert memory_mib["ratio"].values[0] != memory_mib["ratio"].values[0]
        peak_num_entries = df_all[df_all["metric"] == "peak_num_entries"]
        assert len(peak_num_entries.index) == 1
        assert peak_num_entries["before"].values[0] == "unused"
        assert peak_num_entries["ratio"].values[0] != peak_num_entries["ratio"].values[0]
        num_entries = df_all[df_all["metric"] == "num_entries"]
        assert len(num_entries.index) == 1
        assert num_entries["before"].values[0] == 0
        assert num_entries["after"].values[0] == 0
        assert num_entries["ratio"].values[0] == 1.0
        num_buckets = df_all[(df_all["metric"] == "num_buckets") & (df_all["before"] != "skipped")]
        assert len(num_buckets.index) == 1
        assert num_buckets["before"].values[0] == 0
        assert num_buckets["after"].values[0] == 16384
        assert num_buckets["ratio"].values[0] == 1000000000.0

    except Exception as e:
        msg = "__aggregate() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_with_negative_factor() -> None:
    """Test factor -0.1."""
    with pytest.raises(ValueError, match="Factor must be positive!"):
        compare(path_base, path_feature, factor=-0.1)


def test_compare_with_invalid_sort_option() -> None:
    """Test invalid sort option."""
    with pytest.raises(ValueError, match="Invalid sort option!"):
        compare(path_base, path_feature, sort="after")


def test_compare_with_factor_zero_point_one(capsys: Any) -> None:
    """Test factor 0.1."""
    try:
        compare(path_base, path_feature, factor=0.1, only_changed=False, sort="ratio", no_split=False)
        captured = capsys.readouterr()
        assert "Benchmarks that have improved:" in captured.out
        assert "Benchmarks that have stayed the same:" in captured.out
        assert "Benchmarks that have worsened:" in captured.out
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_with_factor_zero_point_three(capsys: Any) -> None:
    """Test factor 0.3."""
    try:
        compare(path_base, path_feature, factor=0.3, only_changed=False, sort="ratio", no_split=False)
        captured = capsys.readouterr()
        assert "Benchmarks that have improved:" in captured.out
        assert "Benchmarks that have stayed the same:" in captured.out
        assert "Benchmarks that have worsened:" in captured.out
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_only_changed(capsys: Any) -> None:
    """Test only changed."""
    try:
        compare(path_base, path_feature, factor=0.2, only_changed=True, sort="ratio", no_split=False)
        captured = capsys.readouterr()
        assert "Benchmarks that have improved:" in captured.out
        assert "Benchmarks that have stayed the same:" not in captured.out
        assert "Benchmarks that have worsened:" in captured.out
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_only_changed_and_no_split(capsys: Any) -> None:
    """Test only changed and no split."""
    try:
        compare(path_base, path_feature, factor=0.2, only_changed=True, sort="ratio", no_split=True)
        captured = capsys.readouterr()
        assert "All changed benchmarks:" in captured.out
        assert "All benchmarks:" not in captured.out
        assert "Benchmarks that have improved:" not in captured.out
        assert "Benchmarks that have stayed the same:" not in captured.out
        assert "Benchmarks that have worsened:" not in captured.out
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_not_only_changed_and_no_split(capsys: Any) -> None:
    """Test only changed and no split."""
    try:
        compare(path_base, path_feature, factor=0.25, only_changed=False, sort="ratio", no_split=True)
        captured = capsys.readouterr()
        assert "All benchmarks:" in captured.out
        assert "All changed benchmarks:" not in captured.out
        assert "Benchmarks that have improved:" not in captured.out
        assert "Benchmarks that have stayed the same:" not in captured.out
        assert "Benchmarks that have worsened:" not in captured.out
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_sort_by_algorithm(capsys: Any) -> None:
    """Test sort by algorithm."""
    try:
        compare(path_base, path_feature, factor=0.2, only_changed=True, sort="algorithm", no_split=True)
        captured = capsys.readouterr()
        assert "All changed benchmarks:" in captured.out
    except Exception as e:
        msg = "compare() should not raise exception!"
        raise AssertionError(msg) from e
