# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for modern PennyLane execution through QDMI."""

# ruff: file-ignore[missing-return-type-private-function]

from __future__ import annotations

import math
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

try:
    import pennylane as qp
except ImportError:
    pytest.skip("Install the PennyLane extra to run these tests.", allow_module_level=True)

from mqt.core.plugins.pennylane import (
    PennyLaneConfigurationError,
    PennyLaneExecutionError,
    PennyLaneValidationError,
    QDMIDevice,
)
from mqt.core.qdmi import Device as QDMIDeviceHandle
from mqt.core.qdmi import Job as QDMIJobHandle
from mqt.core.qdmi import ProgramFormat

from .helpers import Program, patch_open_device, rotation_results, stub_device


def test_uses_already_open_qdmi_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse a device session selected outside the PennyLane adapter."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    monkeypatch.setattr(
        "mqt.core.plugins.pennylane.device.open_device",
        lambda *_args, **_kwargs: pytest.fail("The adapter reopened the QDMI device."),
    )

    device = QDMIDevice(device=cast("QDMIDeviceHandle", qdmi), wires=2)

    assert device.qdmi_device is qdmi
    assert device.device_id is None


def test_samples_counts_probabilities_expectations_and_variances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconstruct common PennyLane result types from raw QDMI samples."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=["left", "right"])

    @qp.qnode(device, shots=100)
    def circuit() -> tuple[object, ...]:
        return (
            qp.sample(wires=["left", "right"]),
            qp.counts(wires=["left", "right"]),
            qp.probs(wires=["left", "right"]),
            qp.expval(qp.PauliZ("left")),
            qp.var(qp.PauliZ("right")),
        )

    samples, counts, probabilities, expectation, variance = circuit()

    assert samples.shape == (100, 2)
    assert counts == {"00": 50, "11": 50}
    np.testing.assert_allclose(probabilities, [0.5, 0.0, 0.0, 0.5])
    assert expectation == pytest.approx(0.0)
    assert variance == pytest.approx(1.0)
    assert device.submitted_jobs == 1
    assert math.isfinite(device.execution_time)


def test_execution_time_accumulates_batch_wall_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count overlapping job execution once for each PennyLane batch."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    readings = iter([0.0, 0.5, 0.5, 1.5, 10.0, 11.0, 11.0, 12.25])
    monkeypatch.setattr("mqt.core.plugins.pennylane.job.monotonic", lambda: next(readings))
    device = QDMIDevice("fake.qdmi", wires=2)

    @qp.qnode(device, shots=[(5, 2), 7])
    def circuit():
        return qp.probs(wires=[0, 1])

    circuit()
    circuit()

    assert device.submitted_jobs == 6
    assert device.execution_time == pytest.approx(1.5 + 2.25)


def test_shot_vectors_submit_before_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Submit every shot-vector copy before waiting for the first job."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)

    @qp.qnode(device, shots=[(5, 2), 7])
    def circuit():
        return qp.probs(wires=[0, 1])

    results = circuit()

    assert len(results) == 3
    assert [submission[2] for submission in qdmi.submissions] == [5, 5, 7]
    assert qdmi.events == ["submit:1", "submit:2", "submit:3", "wait:1", "wait:2", "wait:3"]
    for probabilities in results:
        assert probabilities.shape == (4,)
        assert np.sum(probabilities) == pytest.approx(1.0)


def test_batches_execute_in_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve batch ordering with one QDMI submission per tape."""

    def basis_state_results(program: Program, shots: int) -> list[str]:
        return ["01" if ("PauliX", [], [0]) in program else "10"] * shots

    qdmi = stub_device(result_factory=basis_state_results)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    tapes = (
        qp.tape.QuantumScript([qp.PauliX(0)], [qp.probs(wires=[0, 1])], shots=6),
        qp.tape.QuantumScript([qp.PauliX(1)], [qp.probs(wires=[0, 1])], shots=6),
    )

    results = qp.execute(tapes, device, diff_method=None)

    assert len(results) == 2
    assert len(qdmi.submissions) == 2
    assert qdmi.events == ["submit:1", "submit:2", "wait:1", "wait:2"]
    np.testing.assert_equal(results, ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]))


def test_execution_failure_preserves_submitted_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collect later jobs and retain successful samples without any cancellation."""
    qdmi = stub_device()
    submit_one = qdmi.submit_job

    def fail_wait() -> bool:
        msg = "wait failed"
        raise RuntimeError(msg)

    def submit(
        program: Program,
        program_format: ProgramFormat,
        num_shots: int,
        **parameters: object,
    ) -> QDMIJobHandle:
        job = submit_one(program, program_format, num_shots, **parameters)
        if job.id == "2":
            monkeypatch.setattr(job, "wait", fail_wait)
        return job

    monkeypatch.setattr(qdmi, "submit_job", submit)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=[2, 3, 4])

    with qp.Tracker(device) as tracker, pytest.raises(PennyLaneExecutionError, match="wait failed"):
        device.execute(tape)

    assert tracker.totals == {"batches": 1, "batch_len": 1, "executions": 3, "shots": 9}
    assert device.last_job is not None
    assert [entry.result is not None for entry in device.last_job.entries] == [True, False, True]

    assert qdmi.events == [
        "submit:1",
        "submit:2",
        "submit:3",
        "wait:1",
        "wait:3",
    ]


def test_parameter_shift_gradient_uses_multiple_qdmi_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Differentiate sampled execution through PennyLane's parameter-shift rule."""
    qdmi = stub_device(qubits=1, result_factory=rotation_results)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=["theta"])

    @qp.qnode(device, shots=4000, diff_method="parameter-shift")
    def circuit(angle: float):
        qp.RY(angle, wires="theta")
        return qp.expval(qp.PauliZ("theta"))

    angle = qp.numpy.array(0.4, requires_grad=True)
    value = circuit(angle)
    gradient = qp.grad(circuit)(angle)

    assert value == pytest.approx(np.cos(0.4), abs=0.01)
    assert gradient == pytest.approx(-np.sin(0.4), abs=0.02)
    # One explicit value call, then one forward and two shifted tapes for grad.
    assert device.submitted_jobs == 4


def test_hamiltonian_and_non_commuting_measurements_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let PennyLane split and aggregate Hamiltonian and non-commuting terms."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    hamiltonian = 0.5 * qp.PauliZ(0) + 0.5 * qp.PauliZ(1)

    @qp.qnode(device, shots=100)
    def circuit():
        return qp.expval(hamiltonian), qp.expval(qp.PauliX(0))

    energy, x_expectation = circuit()

    assert energy == pytest.approx(0.0)
    assert np.isfinite(x_expectation)
    assert device.submitted_jobs >= 2


def test_rejects_analytic_execution_before_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject analytic tapes before a QDMI job is created."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)

    @qp.qnode(device)
    def circuit():
        return qp.expval(qp.PauliZ(0))

    with pytest.raises(PennyLaneValidationError, match="finite number of shots"):
        circuit()
    tape = qp.tape.QuantumScript([], [qp.sample(wires=0)], shots=None)
    with pytest.raises(PennyLaneValidationError, match="finite number of shots"):
        device.execute(tape)
    assert not qdmi.submissions


def test_validates_configuration_and_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unknown QDMI parameters and excessive wire counts."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)

    with pytest.raises(PennyLaneConfigurationError, match="unknown"):
        QDMIDevice(
            "fake.qdmi",
            wires=2,
            session_parameters={"unknown": "value"},  # ty: ignore[invalid-argument-type, invalid-key]
        )
    with pytest.raises(PennyLaneConfigurationError, match="3 wires"):
        QDMIDevice("fake.qdmi", wires=3)
    with pytest.raises(PennyLaneConfigurationError, match="exactly one"):
        QDMIDevice()
    with pytest.raises(PennyLaneConfigurationError, match="exactly one"):
        QDMIDevice("fake.qdmi", device=cast("QDMIDeviceHandle", qdmi))
    with pytest.raises(PennyLaneConfigurationError, match="session_parameters"):
        QDMIDevice(
            device=cast("QDMIDeviceHandle", qdmi),
            session_parameters={"token": "unused"},
        )


def test_forwards_job_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward generic QDMI custom job parameters unchanged."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice(
        "fake.qdmi",
        wires=2,
        job_parameters={"custom1": "bucket", "custom2": "prefix"},
    )

    @qp.qnode(device, shots=4)
    def circuit():
        return qp.sample(wires=[0, 1])

    circuit()
    assert qdmi.submissions[0][3] == {"custom1": "bucket", "custom2": "prefix"}


@pytest.mark.parametrize("method", ["one-shot", "tree-traversal"])
def test_rejects_unsupported_mid_circuit_methods(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    """Do not silently replace a requested native MCM method with deferral."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice(device=cast("QDMIDeviceHandle", qdmi), wires=2)

    @qp.qnode(device, shots=5, mcm_method=method)
    def circuit():
        return qp.sample(wires=0)

    with pytest.raises(qp.exceptions.QuantumFunctionError, match="unsupported by the device"):
        circuit()
    assert not qdmi.submissions


def test_rejects_deferred_measurement_without_spare_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail before submission when resetting a measured wire needs an ancilla."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice(device=cast("QDMIDeviceHandle", qdmi), wires=["a", "b"])

    @qp.qnode(device, shots=5)
    def circuit():
        measured = qp.measure("a", reset=True)
        qp.cond(measured, qp.X)("b")
        return qp.sample(wires=["b", "a"])

    with pytest.raises(PennyLaneValidationError, match="require more wires"):
        circuit()
    assert not qdmi.submissions


def test_rejects_postselection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not submit a postselected deferred-measurement circuit."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)

    @qp.qnode(device, shots=5)
    def circuit():
        qp.measure(0, postselect=0)
        return qp.sample(wires=1)

    with pytest.raises(ValueError, match="postselection"):
        circuit()
    assert not qdmi.submissions


def test_qnode_shots_and_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Track actual jobs and shot copies while QNode shots specify execution budgets."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice(device=cast("QDMIDeviceHandle", qdmi), wires=2)

    @qp.qnode(device, shots=7)
    def circuit():
        return qp.sample(wires=[0, 1])

    with qp.Tracker(device) as tracker:
        assert circuit().shape == (7, 2)
        results = qp.set_shots(circuit, shots=[(3, 2), 5])()
    assert [result.shape for result in results] == [(3, 2), (3, 2), (5, 2)]
    assert tracker.totals == {"batches": 2, "batch_len": 2, "executions": 4, "shots": 18}
    assert device.submitted_jobs == 4
    assert device.shots.total_shots is None
    circuit()
    assert tracker.totals["executions"] == 4


@pytest.mark.parametrize("max_retries", [None, 3])
def test_retry_shot_copies_preserves_order_and_tracking(
    monkeypatch: pytest.MonkeyPatch, max_retries: int | None
) -> None:
    """Replace one failed shot copy without repeating successful copies or losing their mapping."""
    qdmi = stub_device()
    original = qdmi.submit_job
    handles = []

    def submit(program: Program, program_format: ProgramFormat, num_shots: int, **parameters: object) -> QDMIJobHandle:
        handle = original(program, program_format, num_shots, **parameters)
        handle = cast("Mock", handle)
        handle.check.side_effect = None
        handle.check.return_value = QDMIJobHandle.Status.FAILED if not handles else QDMIJobHandle.Status.DONE
        handles.append(handle)
        return handle

    monkeypatch.setattr(qdmi, "submit_job", submit)
    patch_open_device(monkeypatch, qdmi)
    if max_retries is None:
        device = QDMIDevice("fake.qdmi", wires=2, job_parameters={"custom1": 9})
    else:
        device = QDMIDevice("fake.qdmi", wires=2, job_parameters={"custom1": 9}, max_retries=max_retries)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=[2, 3, 4])
    with qp.Tracker(device) as tracker:
        if max_retries:
            result = device.execute(tape)
        else:
            with pytest.raises(PennyLaneExecutionError) as caught:
                device.execute(tape)
            assert caught.value.job is device.last_job
            assert device.last_job is not None
            assert device.submitted_jobs == 3
            device.last_job.resubmit([0])
            result = device.last_job.result()
    assert [samples.shape for samples in result] == [(2, 2), (3, 2), (4, 2)]
    assert [submission[2] for submission in qdmi.submissions] == [2, 3, 4, 2]
    assert qdmi.submissions[0] == qdmi.submissions[3]
    assert device.submitted_jobs == 4
    assert tracker.totals == {"batches": 1, "batch_len": 1, "executions": 4, "shots": 11}
    assert device.last_job is not None
    assert [(entry.input_index, entry.shot_index) for entry in device.last_job.entries] == [(0, 0), (0, 1), (0, 2)]
    assert device.last_job.entries[0].automatic_retries == int(bool(max_retries))
    for handle in handles[1:]:
        handle.get_shots.assert_called_once()
        handle.cancel.assert_not_called()


@pytest.mark.parametrize("value", [-1, True, 1.5, "3", None])
def test_invalid_retry_configuration(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    """Reject invalid retry counts through both PennyLane constructors before execution."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    with pytest.raises(PennyLaneConfigurationError, match="max_retries"):
        QDMIDevice("fake.qdmi", max_retries=cast("int", value))
    with pytest.raises(PennyLaneConfigurationError, match="max_retries"):
        qp.device("mqt.ddsim.default", max_retries=value)
    assert not qdmi.submissions


@pytest.mark.parametrize("interrupted", [False, True])
def test_partial_submission_retains_pennylane_batch(monkeypatch: pytest.MonkeyPatch, *, interrupted: bool) -> None:
    """Submission exceptions and interruptions retain handles for accepted shot copies."""
    qdmi = stub_device()
    original = qdmi.submit_job
    cause = KeyboardInterrupt() if interrupted else RuntimeError("submission failed")

    def submit(program: Program, program_format: ProgramFormat, num_shots: int, **parameters: object) -> QDMIJobHandle:
        if len(qdmi.submissions) == 1:
            raise cause
        return original(program, program_format, num_shots, **parameters)

    monkeypatch.setattr(qdmi, "submit_job", submit)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=[2, 3, 4])
    with pytest.raises(KeyboardInterrupt if interrupted else PennyLaneExecutionError) as caught:
        device.execute(tape)
    batch = device.last_job
    assert batch is not None
    if isinstance(caught.value, PennyLaneExecutionError):
        assert caught.value.job is batch
    batch.collect()
    monkeypatch.setattr(qdmi, "submit_job", original)
    batch.resubmit([1], allow_unknown=True)
    batch.submit()
    assert [samples.shape for samples in batch.result()] == [(2, 2), (3, 2), (4, 2)]
    assert device.submitted_jobs == 3
    assert not any(event.startswith("cancel") for event in qdmi.events)


def test_preparation_interruption_clears_last_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted new execution cannot expose a preceding batch as its recovery handle."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=1)
    samples = device.execute(tape)
    previous = device.last_job
    assert previous is not None

    def interrupt(*_args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("mqt.core.plugins.pennylane.device.QDMIDevice._prepare", interrupt)
    with pytest.raises(KeyboardInterrupt):
        device.execute(qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=2))
    assert device.last_job is None
    np.testing.assert_array_equal(previous.result(), samples)
    assert len(qdmi.submissions) == 1


def test_result_failure_retains_native_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain native result errors in the recoverable batch."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    failure = RuntimeError("shots unavailable; counts unavailable")
    monkeypatch.setattr("mqt.core.plugins.pennylane.job.PennyLaneJob._samples", Mock(side_effect=failure))
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=2)
    with pytest.raises(PennyLaneExecutionError) as caught:
        device.execute(tape)
    assert caught.value.job is not None
    recorded = caught.value.job.entries[0].attempts[0].failures[0]
    assert recorded.stage == "result"
    assert recorded.cause is failure
    assert len(qdmi.submissions) == 1


def test_tracking_interruption_keeps_accepted_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tracker callback cannot lose a job that the device already accepted."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=[2, 3])

    def callback(**_kwargs: object) -> None:
        if device.submitted_jobs:
            raise KeyboardInterrupt

    with qp.Tracker(device, callback=callback), pytest.raises(KeyboardInterrupt):
        device.execute(tape)
    batch = device.last_job
    assert batch is not None
    assert batch.entries[0].attempts[0].handle is not None
    assert not batch.entries[1].attempts
    batch.submit([1])
    assert [samples.shape for samples in batch.result()] == [(2, 2), (3, 2)]
    assert device.submitted_jobs == 2


def test_native_groups_preserve_heterogeneous_shot_partitions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Group equal shots without changing tape order, custom parameters, or tracking."""
    qdmi = stub_device()
    groups = []

    def submit(programs: list[Program], program_format: ProgramFormat, shots: int, **parameters: object):
        handle = Mock()
        handle.wait.return_value = True
        handle.check.return_value = QDMIJobHandle.Status.DONE
        handle.get_program_status.return_value = None
        handle.get_shots.side_effect = lambda program_index: [str(program_index) * 2] * shots
        groups.append((programs, program_format, shots, parameters, handle))
        return handle

    monkeypatch.setattr(qdmi, "try_submit_job", submit)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice(device=cast("QDMIDeviceHandle", qdmi), wires=2, job_parameters={"custom1": 42})
    tapes = (
        qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=[2, 3, 2]),
        qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=3),
    )
    with qp.Tracker(device) as tracker:
        results = qp.execute(tapes, device, diff_method=None)
    assert [group[2] for group in groups] == [2, 3]
    assert all(len(group[0]) == 2 and group[3] == {"custom1": 42} for group in groups)
    assert not qdmi.submissions
    assert device.submitted_jobs == 2
    assert tracker.totals["executions"] == 4
    assert tracker.totals["shots"] == 10
    for result, expected in zip(
        (*results[0], results[1]), (np.zeros((2, 2)), np.zeros((3, 2)), np.ones((2, 2)), np.ones((3, 2))), strict=True
    ):
        np.testing.assert_array_equal(result, expected)
    for *_, handle in groups:
        handle.check.assert_called_once()
        handle.wait.assert_called_once()
