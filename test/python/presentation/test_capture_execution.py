# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Execution capture consistency checks for the MQSF presentation."""

from __future__ import annotations

import importlib.util
import json
import runpy
import subprocess
import sys
import unittest
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "presentations/mqsf2026/capture_execution.py"


def test_failed_later_export_preserves_previous_capture(tmp_path: Path) -> None:
    """A failed required worker must fail the CLI without publishing earlier successes."""
    input_path = tmp_path / "programs.json"
    input_path.write_text(
        json.dumps({
            "provenance": {},
            "scenarios": [
                {
                    "id": "example",
                    "variants": [
                        {
                            "id": "structured",
                            "exports": [{"id": name, "code": name} for name in ("openqasm3", "qir-adaptive")],
                        }
                    ],
                }
            ],
        })
    )
    output_path = tmp_path / "demo.json.gz"
    output_path.write_bytes(b"previous verified capture")
    # Run the real CLI with two controlled native worker outcomes.
    harness = """
import runpy, subprocess, sys
from unittest.mock import patch
sys.argv = sys.argv[1:]
with patch("subprocess.run", side_effect=[
    subprocess.CompletedProcess([], 0, stdout='{"duration_ms": 1}'),
    subprocess.CalledProcessError(1, "native worker", stderr="execution failed"),
]):
    runpy.run_path(sys.argv[0], run_name="__main__")
"""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - exercise the local capture CLI
        [
            sys.executable,
            "-c",
            harness,
            str(SCRIPT),
            "--library",
            str(tmp_path / "unused.so"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Captured example/structured/openqasm3" in result.stdout
    assert output_path.read_bytes() == b"previous verified capture"


def test_shot_timing_matches_results_and_measured_execution_window() -> None:
    """Reject invented order, future timestamps, and outcome mismatches."""
    validate = runpy.run_path(str(SCRIPT))["validate_execution"]
    execution = {
        "shots": ["00", "11"],
        "counts": {"00": 1, "11": 1},
        "num_shots": 2,
        "terminal_status": "DONE",
        "payload_identity_verified": True,
        "events": [{"time_ms": 1.0}],
        "submitted_ms": 2.0,
        "completed_ms": 5.0,
        "shot_events": [
            {"time_ms": 3.0, "shot_index": 0, "outcome": "00"},
            {"time_ms": 4.0, "shot_index": 1, "outcome": "11"},
        ],
    }
    validate(execution)
    validate(execution | {"runtime_events": [{"time_ms": 2.5}, {"time_ms": 4.5}]})
    for times in ([4.5, 2.5], [1.5], [5.5]):
        with pytest.raises(ValueError, match="Runtime observations"):
            validate(execution | {"runtime_events": [{"time_ms": time} for time in times]})
    for invalid in ({"time_ms": 6.0}, {"outcome": "00"}, {"shot_index": 0}):
        shots = [execution["shot_events"][0], execution["shot_events"][1] | invalid]
        with pytest.raises(ValueError, match="Shot completion"):
            validate(execution | {"shot_events": shots})


class ExecutionCaptureTest(unittest.TestCase):
    """Keep shot order, returned counts, and native evidence consistent."""

    def test_rejects_inconsistent_execution_evidence(self) -> None:
        """Reject independent evidence mismatches rather than accepting plausible counts."""
        script = Path(__file__).resolve().parents[3] / "presentations/mqsf2026/capture_execution.py"
        assert script.is_file(), "The execution capture is not implemented"
        spec = importlib.util.spec_from_file_location("capture_execution", script)
        assert spec is not None
        assert spec.loader is not None
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        execution = {
            "shots": ["10", "00", "10"],
            "counts": {"10": 2, "00": 1},
            "num_shots": 3,
            "terminal_status": "DONE",
            "payload_identity_verified": True,
            "events": [
                {"time_ms": 0.0, "status": "QDMI_SUCCESS"},
                {"time_ms": 1.0, "status": "QDMI_SUCCESS"},
            ],
        }
        capture.validate_execution(execution)
        for field, invalid in (
            ("counts", {"10": 1, "00": 2}),
            ("terminal_status", "FAILED"),
            ("payload_identity_verified", False),
            ("payload_identity_verified", "false"),
            ("num_shots", 4),
        ):
            with (
                self.subTest(field=field),
                pytest.raises(ValueError, match=r"Execution must|Returned shot count|Ordered shots disagree"),
            ):
                capture.validate_execution(execution | {field: invalid})


if __name__ == "__main__":
    unittest.main()
