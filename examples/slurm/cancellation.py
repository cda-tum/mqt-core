# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# This helper is copied beside workloads, not installed as a package.
# ruff: file-ignore[implicit-namespace-package]

"""Opt-in cancellation for a QDMI workload launched by Slurm."""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType

    from mqt.core.qdmi import Job


def run_job(submit: Callable[[], Job]) -> Job:
    """Submit and wait for one job on the main thread, cancelling it on SIGTERM.

    The handler is active during submission and restored on exit. Provider errors
    propagate to the caller. Submission and cancellation themselves may block.

    Returns:
        The completed job.

    Raises:
        SystemExit: With status 143 after requesting cancellation.
    """
    terminated = False

    def terminate(_signum: int, _frame: FrameType | None) -> None:
        nonlocal terminated
        terminated = True

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        job = submit()
        try:
            while not terminated:
                if job.wait(1):
                    break
        finally:
            if terminated:
                job.cancel()
        if terminated:
            raise SystemExit(128 + signal.SIGTERM)
        return job
    finally:
        signal.signal(signal.SIGTERM, previous)
