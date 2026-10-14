# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Protect the dashboard's interpretation of saved asv measurements."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from benchmarks.dashboard import METRICS, build_dashboard, read_results

if TYPE_CHECKING:
    from pathlib import Path


def write_results(root: Path, *, values: tuple[float | None, float | None] = (0, 12)) -> dict[str, Any]:
    """Write an asv database with reordered columns and omitted optional fields.

    Returns:
        The result record for further test-specific modifications.
    """
    root.mkdir()
    machine = root / "test-machine"
    machine.mkdir()
    (machine / "machine.json").write_text(json.dumps({"machine": "test-machine", "cpu": "test CPU", "version": 1}))
    params = [["'line'"], ["4", "8"], ["'ghz'"]]
    (root / "benchmarks.json").write_text(
        json.dumps({
            "version": 2,
            **{
                metric["benchmark"]: {
                    "param_names": ["topology", "qubits", "circuit"],
                    "params": params,
                    "version": "current",
                }
                for metric in METRICS
            },
        })
    )
    result = {
        "version": 2,
        "commit_hash": "a" * 40,
        "date": 2000,
        "env_name": "uv-py3.13-qiskit2.5.2",
        "python": "3.13",
        "requirements": {"qiskit": "2.5.2"},
        "result_columns": ["params", "version", "result", "started_at", "samples", "profile"],
        "results": {
            metric["benchmark"]: [params, "current", list(values), 3000, [[0, 0], [11, 13]]] for metric in METRICS
        },
    }
    (machine / "run.json").write_text(json.dumps(result))
    return result


def test_decodes_column_names_and_parameter_order(tmp_path: Path) -> None:
    """Column and parameter ordering must not silently swap metrics or cases."""
    write_results(tmp_path / "results")
    run = read_results(tmp_path / "results")["runs"][0]
    assert run["measured_at"] == 3000
    assert [(row["circuit"], row["qubits"], row["topology"]) for row in run["rows"]] == [
        ("ghz", 4, "line"),
        ("ghz", 8, "line"),
    ]
    assert run["rows"][0]["depth"]["value"] == 0
    assert run["rows"][0]["depth"]["status"] == "ok"
    assert run["rows"][1]["time"]["samples"] == [11, 13]


@pytest.mark.parametrize("invalid", [None, -1, float("nan"), float("inf")])
def test_unavailable_values_never_become_zero(tmp_path: Path, invalid: float | None) -> None:
    """Failed and invalid readings must not appear to improve performance."""
    write_results(tmp_path / "results", values=(invalid, 12))
    cell = read_results(tmp_path / "results")["runs"][0]["rows"][0]["time"]
    assert cell["value"] is None
    assert cell["status"] == "unavailable"


def test_missing_metrics_and_old_versions_remain_visible(tmp_path: Path) -> None:
    """An incomplete run keeps its cases, and old versions remain identifiable."""
    root = tmp_path / "results"
    result = write_results(root)
    del result["results"][METRICS[0]["benchmark"]]
    result["results"][METRICS[1]["benchmark"]][1] = "older-suite"
    (root / "test-machine/run.json").write_text(json.dumps(result))
    row = read_results(root)["runs"][0]["rows"][0]
    assert row["time"]["value"] is None
    assert row["two_qubit_count"]["version"] == "older-suite"
    assert row["two_qubit_count"]["current_suite"] is False


def test_orders_revisions_and_keeps_environments_separate(tmp_path: Path) -> None:
    """Commit date sets history order; distinct environments are not merged."""
    root = tmp_path / "results"
    result = write_results(root)
    result.update(commit_hash="b" * 40, date=1000, env_name="another-environment")
    (root / "test-machine/other.json").write_text(json.dumps(result))
    runs = read_results(root)["runs"]
    assert [run["commit"] for run in runs] == ["a" * 40, "b" * 40]
    assert runs[0]["environment"] != runs[1]["environment"]


def test_unknown_result_schema_fails_build(tmp_path: Path) -> None:
    """Unknown asv formats require a decoder update, not guessed values."""
    root = tmp_path / "results"
    result = write_results(root)
    result["version"] = 99
    (root / "test-machine/run.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Unsupported asv results version"):
        read_results(root)


def test_build_keeps_asv_details_and_exports_data(tmp_path: Path) -> None:
    """Overview publication preserves asv's site and exports the decoded samples."""
    root = tmp_path / "results"
    write_results(root)
    output = tmp_path / "html"
    (output / "asv").mkdir(parents=True)
    (output / "asv/index.html").write_text("original asv site")
    build_dashboard(root, output)
    assert (output / "asv/index.html").read_text() == "original asv site"
    assert {"index.html", "dashboard.js", "dashboard.css", "measurements.json"} <= {
        path.name for path in output.iterdir()
    }
    exported = json.loads((output / "measurements.json").read_text())
    assert len(exported["runs"][0]["rows"]) == 2
    assert exported["runs"][0]["rows"][1]["time"]["samples"] == [11, 13]


def test_sphinx_fragment_uses_version_relative_assets(tmp_path: Path) -> None:
    """Embedded assets stay under the RTD version and outside the public fragment."""
    results = tmp_path / "results"
    write_results(results)
    output = tmp_path / "public/compiler-benchmarks"
    fragment = tmp_path / "fragment.html"
    build_dashboard(results, output, embed_path=fragment)
    html = fragment.read_text()
    assert 'src="compiler-benchmarks/dashboard.js"' in html
    assert 'href="compiler-benchmarks/measurements.json"' in html
    assert 'class="mqt-dashboard" data-embedded' in html
    assert not (output / "embed.html").exists()
