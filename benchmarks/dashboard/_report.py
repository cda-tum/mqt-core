# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Decode asv results and build the static MQT overview."""

from __future__ import annotations

import ast
import itertools
import json
import math
import operator
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METRICS = (
    {"id": "time", "benchmark": "compiler.Compilation.time_compile", "label": "Compilation time", "unit": "s"},
    {
        "id": "two_qubit_count",
        "benchmark": "compiler.CircuitQuality.track_two_qubit_count",
        "label": "Two-qubit gates",
        "unit": "gates",
    },
    {"id": "depth", "benchmark": "compiler.CircuitQuality.track_depth", "label": "Circuit depth", "unit": "layers"},
)


def read_results(results_dir: Path) -> dict[str, Any]:
    """Decode asv v2 results, retaining failures, versions, and timing samples.

    Returns:
        Machines, metric definitions, and runs with one row per circuit case.

    Raises:
        ValueError: If the database uses an unsupported schema or parameter set.
    """
    current_metadata = json.loads((results_dir / "benchmarks.json").read_text())
    runs: list[dict[str, Any]] = []
    machines = {}
    for machine_file in sorted(results_dir.glob("*/machine.json")):
        machine = json.loads(machine_file.read_text())
        machines[machine["machine"]] = machine
        snapshot = machine_file.with_name("benchmarks.json")
        metadata = json.loads(snapshot.read_text()) if snapshot.exists() else current_metadata
        for result_file in sorted(machine_file.parent.glob("*.json")):
            if result_file in {machine_file, snapshot}:
                continue
            result = json.loads(result_file.read_text())
            if result.get("version") != 2:
                msg = f"Unsupported asv results version in {result_file}"
                raise ValueError(msg)
            rows: dict[tuple[Any, ...], dict[str, Any]] = {}
            started = []
            for metric in METRICS:
                definition = metadata[metric["benchmark"]]
                names = definition["param_names"]
                if set(names) != {"circuit", "qubits", "topology"}:
                    msg = f"Unsupported parameters for {metric['benchmark']}: {names}"
                    raise ValueError(msg)
                saved = result["results"].get(metric["benchmark"], [])
                columns = dict(zip(result["result_columns"], saved, strict=False))
                if columns.get("started_at") is not None:
                    started.append(columns["started_at"])
                params = columns.get("params", definition["params"])
                for index, values in enumerate(itertools.product(*params)):
                    case = dict(zip(names, (ast.literal_eval(value) for value in values), strict=True))
                    key = (case["circuit"], case["qubits"], case["topology"])
                    row = rows.setdefault(key, case)
                    readings = columns.get("result") or []
                    value = readings[index] if index < len(readings) else None
                    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
                    valid = valid and math.isfinite(value) and value >= 0
                    cell: dict[str, Any] = {
                        "value": value if valid else None,
                        "status": "ok" if valid else "unavailable",
                        "version": columns.get("version"),
                        "current_suite": columns.get("version") == definition["version"],
                        "input_version": (machine.get("measurement") or {})
                        .get("input_versions", {})
                        .get(result["commit_hash"]),
                    }
                    for source, destination in (
                        ("stats_q_25", "q25"),
                        ("stats_q_75", "q75"),
                        ("stats_repeat", "repeat"),
                        ("samples", "samples"),
                    ):
                        entries = columns.get(source) or []
                        cell[destination] = entries[index] if index < len(entries) else None
                    row[metric["id"]] = cell
            if not rows:
                continue
            for row in rows.values():
                for metric in METRICS:
                    row.setdefault(metric["id"], {"value": None, "status": "unavailable", "version": None})
            runs.append({
                "commit": result["commit_hash"],
                "date": result["date"],
                "measured_at": max(started, default=None),
                "machine": machine["machine"],
                "measurement": machine.get("measurement"),
                "environment": result["env_name"],
                "python": result["python"],
                "requirements": result["requirements"],
                "rows": sorted(rows.values(), key=operator.itemgetter("circuit", "qubits", "topology")),
            })
    return {
        "schema": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": METRICS,
        "machines": machines,
        "batches": [json.loads(path.read_text()) for path in sorted((results_dir / "batches").glob("*.json"))],
        "runs": sorted(
            runs,
            key=lambda run: (
                run["measurement"]["created_at"] if run["measurement"] else "",
                run["date"],
                run["commit"],
            ),
            reverse=True,
        ),
    }


def build_dashboard(results_dir: Path, output_dir: Path, *, embed_path: Path | None = None) -> None:
    """Write the MQT overview alongside the unmodified asv detail site."""
    data = read_results(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "dashboard.css", "dashboard.js"):
        shutil.copyfile(Path(__file__).with_name(name), output_dir / name)
    source = Path(__file__).with_name("index.html").read_text()
    fragment = source.split("<!-- dashboard:start -->", 1)[1].split("<!-- dashboard:end -->", 1)[0]
    fragment = fragment.replace('href="asv/', 'href="compiler-benchmarks/asv/')
    fragment = fragment.replace('href="measurements.json"', 'href="compiler-benchmarks/measurements.json"')
    if embed_path is not None:
        embed_path.parent.mkdir(parents=True, exist_ok=True)
        embed_path.write_text(
            '<link rel="stylesheet" href="compiler-benchmarks/dashboard.css">\n'
            '<div class="mqt-dashboard" data-embedded>\n' + fragment + "\n</div>\n"
            '<script src="compiler-benchmarks/dashboard.js" defer></script>\n'
        )
    (output_dir / "measurements.json").write_text(json.dumps(data, allow_nan=False, indent=2) + "\n")
