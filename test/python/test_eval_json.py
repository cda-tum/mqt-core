"""Test the evaluation of JSON results."""

from __future__ import annotations

from math import inf
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mqt.core.evaluation import __aggregate, __flatten_dict, __post_processing

if TYPE_CHECKING:
    from pytest_console_scripts import ScriptRunner

path_base = Path(__file__).resolve().parent / "results_baseline.json"
path_feature = Path(__file__).resolve().parent / "results_feature.json"


def test_flatten_dict() -> None:
    """Test flatten_dict."""
    d1 = {"a": {"b": {"main": 1}}}
    assert __flatten_dict(d1) == {"a.b.main": 1}
    d2 = {"a": {"b": {"main": 1, "feature": 2}}, "d": {"main": 2}}
    assert __flatten_dict(d2) == {"a.b.main": 1, "a.b.feature": 2, "d.main": 2}


def test_post_processing() -> None:
    """Test postprocessing."""
    with pytest.raises(ValueError, match="Benchmark a.b is missing algorithm, task, number of qubits or metric!"):
        __post_processing("a.b")
    assert __post_processing("BV.Functionality.1024.runtime") == {
        "algorithm": "BV",
        "task": "Functionality",
        "num_qubits": "1024",
        "component": "",
        "metric": "runtime",
    }
    assert __post_processing("GHZ.Simulation.128.dd.active_memory_mib") == {
        "algorithm": "GHZ",
        "task": "Simulation",
        "num_qubits": "128",
        "component": "",
        "metric": "active_memory_mib",
    }
    assert __post_processing("RandomClifford.Simulation.14.dd.real_numbers.cache_manager.memory_used_MiB_peak") == {
        "algorithm": "RandomClifford",
        "task": "Simulation",
        "num_qubits": "14",
        "component": "real_numbers_cache_manager",
        "metric": "memory_used_MiB_peak",
    }
    assert __post_processing("QPE.Functionality.15.dd.matrix.unique_table.total.lookups") == {
        "algorithm": "QPE",
        "task": "Functionality",
        "num_qubits": "15",
        "component": "matrix_unique_table",
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
        hit_ratio = df_all[df_all["metric"] == "hit_ratio*"]
        assert len(hit_ratio.index) == 1
        assert hit_ratio["before"].values[0] == 0.5
        assert hit_ratio["after"].values[0] == 0.8
        assert hit_ratio["ratio"].values[0] == 0.625
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
        assert num_buckets["ratio"].values[0] == inf

    except Exception as e:
        msg = "__aggregate() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_with_negative_factor(script_runner: ScriptRunner) -> None:
    """Test factor -0.1."""
    ret = script_runner.run([
        "compare",
        "./test/python/results_baseline.json",
        "./test/python/results_feature.json",
        "--factor=-0.1",
    ])
    assert not ret.success
    assert "Factor must be positive!" in ret.stderr


@pytest.mark.script_launch_mode("subprocess")
def test_compare_with_invalid_sort_option(script_runner: ScriptRunner) -> None:
    """Test invalid sort option."""
    ret = script_runner.run([
        "compare",
        "./test/python/results_baseline.json",
        "./test/python/results_feature.json",
        "--sort=after",
    ])
    assert not ret.success
    assert "Invalid sort option!" in ret.stderr


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_default_parameters(script_runner: ScriptRunner) -> None:
    """Testing the command line functionality with default parameters."""
    ret = script_runner.run([
        "compare",
        "./test/python/results_baseline.json",
        "./test/python/results_feature.json",
    ])
    assert "DD runtimes:" in ret.stdout
    assert "Runtimes that have improved:" in ret.stdout
    assert "Runtimes that have stayed the same:" in ret.stdout
    assert "Runtimes that have worsened:" in ret.stdout
    assert "DD details:" not in ret.stdout
    assert "DD Benchmarks that have improved:" not in ret.stdout
    assert "DD Benchmarks that have stayed the same:" not in ret.stdout
    assert "DD Benchmarks that have worsened:" not in ret.stdout
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_factor_point_three(script_runner: ScriptRunner) -> None:
    """Testing the command line functionality with default parameters, except that factor is set to 0.3 and dd details should be shown."""
    ret = script_runner.run([
        "compare",
        "./test/python/results_baseline.json",
        "./test/python/results_feature.json",
        "--factor=0.3",
        "--dd",
    ])
    assert "DD runtimes:" in ret.stdout
    assert "Runtimes that have improved:" in ret.stdout
    assert "Runtimes that have stayed the same:" in ret.stdout
    assert "Runtimes that have worsened:" in ret.stdout
    assert "DD details:" in ret.stdout
    assert "DD Benchmarks that have improved:" in ret.stdout
    assert "DD Benchmarks that have stayed the same:" in ret.stdout
    assert "DD Benchmarks that have worsened:" in ret.stdout
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_only_changed(script_runner: ScriptRunner) -> None:
    """Testing the command line functionality with default parameters, except that factor is set to 0.2 and only_changed is set to true."""
    ret = script_runner.run([
        "compare",
        "./test/python/results_baseline.json",
        "./test/python/results_feature.json",
        "--factor=0.2",
        "--only_changed",
    ])
    assert "Runtimes that have improved:" in ret.stdout
    assert "Runtimes that have stayed the same:" not in ret.stdout
    assert "Runtimes that have worsened:" in ret.stdout
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_only_changed_and_no_split(script_runner: ScriptRunner) -> None:
    """Testing the command line functionality with factor=0.1 per default, but both only_changed and no_split are set to true."""
    ret = script_runner.run([
        "compare",
        "./test/python/results_baseline.json",
        "./test/python/results_feature.json",
        "--dd",
        "--only_changed",
        "--no_split",
    ])
    assert "All changed runtimes:" in ret.stdout
    assert "All runtimes:" not in ret.stdout
    assert "Runtimes that have improved:" not in ret.stdout
    assert "Runtimes that have stayed the same:" not in ret.stdout
    assert "Runtimes that have worsened:" not in ret.stdout
    assert "All changed DD benchmarks:" in ret.stdout
    assert "All DD benchmarks:" not in ret.stdout
    assert "DD Benchmarks that have improved:" not in ret.stdout
    assert "DD Benchmarks that have stayed the same:" not in ret.stdout
    assert "DD Benchmarks that have worsened:" not in ret.stdout
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_no_split(script_runner: ScriptRunner) -> None:
    """Testing the command line functionality with default parameters, except for no_split set to true."""
    ret = script_runner.run([
        "compare",
        "./test/python/results_baseline.json",
        "./test/python/results_feature.json",
        "--no_split",
        "--dd",
    ])
    assert "All runtimes:" in ret.stdout
    assert "All changed runtimes:" not in ret.stdout
    assert "All DD benchmarks:" in ret.stdout
    assert "All DD changed benchmarks:" not in ret.stdout
    assert "DD Benchmarks that have improved:" not in ret.stdout
    assert "DD Benchmarks that have stayed the same:" not in ret.stdout
    assert "DD Benchmarks that have worsened:" not in ret.stdout
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_sort_by_algorithm(script_runner: ScriptRunner) -> None:
    """Testing the command line functionality with sort set with "algorithm" and only_changed and no_split set to true."""
    ret = script_runner.run([
        "compare",
        "./test/python/results_baseline.json",
        "./test/python/results_feature.json",
        "--sort=algorithm",
        "--only_changed",
        "--no_split",
    ])
    assert "All changed runtimes:" in ret.stdout
    assert ret.success
