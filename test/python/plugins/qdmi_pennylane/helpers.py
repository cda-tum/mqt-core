# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Mock the native preparation boundary for PennyLane scheduling tests."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import numpy as np

from mqt.core.qdmi import Device as QDMIDevice
from mqt.core.qdmi import Job as QDMIJob
from mqt.core.qdmi import ProgramFormat

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pytest

Program = list[tuple[str, list[float], list[int]]]


def bell_results(_program: Program, shots: int) -> list[str]:
    """Return an even Bell-state histogram."""
    return ["00"] * (shots // 2) + ["11"] * (shots - shots // 2)


def rotation_results(program: Program, shots: int) -> list[str]:
    """Return deterministic outcomes for a prepared RY rotation."""
    angle = next(parameters[0] for name, parameters, _ in program if name == "RY")
    ones = round(shots * math.sin(angle / 2) ** 2)
    return ["0"] * (shots - ones) + ["1"] * ones


class StubDevice:
    """Record accepted jobs and control their completion for scheduler tests."""

    def __init__(
        self,
        *,
        qubits: int = 2,
        result_factory: Callable[[Program, int], Sequence[str]] = bell_results,
    ) -> None:
        """Store deterministic readout behavior."""
        self._qubits = qubits
        self._result_factory = result_factory
        self.submissions: list[tuple[Program, ProgramFormat, int, Mapping[str, object]]] = []
        self.events: list[str] = []

    @staticmethod
    def name() -> str:
        """Return the stub name."""
        return "stub.qdmi"

    def qubits_num(self) -> int:
        """Return the device width."""
        return self._qubits

    def try_submit_job(self, *_args: object, **_kwargs: object) -> None:
        """Fall back to independent jobs unless a test supplies native grouping."""

    def submit_job(
        self, program: Program, program_format: ProgramFormat, num_shots: int, **parameters: object
    ) -> QDMIJob:
        """Record a submission and return a completed job mock.

        Returns:
            The accepted job handle.
        """
        self.submissions.append((program, program_format, num_shots, parameters))
        job = Mock()
        job.get_program_status.return_value = None
        job.id = str(len(self.submissions))
        self.events.append(f"submit:{job.id}")

        def wait() -> bool:
            self.events.append(f"wait:{job.id}")
            return True

        job.wait.side_effect = wait
        job.cancel.side_effect = lambda: self.events.append(f"cancel:{job.id}")
        job.check.return_value = QDMIJob.Status.DONE
        job.get_shots.return_value = list(self._result_factory(program, num_shots))
        return cast("QDMIJob", job)


def stub_device(
    *, qubits: int = 2, result_factory: Callable[[Program, int], Sequence[str]] = bell_results
) -> StubDevice:
    """Return a scheduler test device."""
    return StubDevice(qubits=qubits, result_factory=result_factory)


def patch_open_device(monkeypatch: pytest.MonkeyPatch, device: StubDevice) -> Mock:
    """Mock opening and native preparation while retaining the real Batch scheduler.

    Returns:
        The mocked native context.
    """
    monkeypatch.setattr("mqt.core.plugins.pennylane.device.open_device", lambda *_args, **_kwargs: device)
    context = Mock(id=1)

    def recorder(_wires: int) -> Mock:
        operations: Program = []
        record = Mock()
        record.named_operation.side_effect = lambda name, parameters, wires, *_modifiers: operations.append((
            name,
            parameters,
            wires,
        ))
        program = Mock(program_format=ProgramFormat.QASM3, operations=operations)
        program.submit.side_effect = lambda shots: device.submit_job(
            operations, ProgramFormat.QASM3, shots, **context.parameters
        )
        program.decode_samples.side_effect = lambda job, _shots, index: np.asarray(
            [[int(bit) for bit in row[::-1]] for row in job.get_shots(index)], dtype=np.int8
        )
        record.prepare.return_value = program
        return record

    def create(_device: QDMIDevice, **parameters: object) -> Mock:
        context.parameters = parameters
        return context

    context.recorder.side_effect = recorder
    context.try_submit_programs.side_effect = lambda programs, shots: device.try_submit_job(
        [program.operations for program in programs], ProgramFormat.QASM3, shots, **context.parameters
    )
    monkeypatch.setattr("mqt.core.plugins.pennylane.device._PennyLaneQDMIContext", create)
    return context
